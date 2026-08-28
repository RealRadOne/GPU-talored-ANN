#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>

// ------- dtype 映射到 numpy descr -------
template <typename T>
struct npy_descr;

template <> struct npy_descr<uint32_t> { static constexpr const char* value = "<u4"; };
template <> struct npy_descr<int32_t>  { static constexpr const char* value = "<i4"; };
template <> struct npy_descr<int64_t>  { static constexpr const char* value = "<i8"; };
template <> struct npy_descr<float>    { static constexpr const char* value = "<f4"; };
template <> struct npy_descr<double>   { static constexpr const char* value = "<f8"; };

inline void require(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error(msg);
}

// 写 2D C-order (row-major) 的 .npy
template <typename T>
void write_npy_2d(const std::string& path, const T* data, int64_t rows, int64_t cols) {
  static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
  require(rows > 0 && cols > 0, "write_npy_2d: rows/cols must be > 0");
  require(data != nullptr, "write_npy_2d: data is null");

  std::ofstream out(path, std::ios::binary);
  require(out.is_open(), "Cannot open for write: " + path);

  // magic + version
  const char magic[] = "\x93NUMPY";
  out.write(magic, 6);
  const uint8_t ver_major = 1, ver_minor = 0;
  out.put(static_cast<char>(ver_major));
  out.put(static_cast<char>(ver_minor));

  // header dict (Python literal), must end with '\n'
  // 注意：shape 必须是 "(rows, cols)"，且逗号和空格格式要对
  std::string header = "{'descr': '";
  header += npy_descr<T>::value;
  header += "', 'fortran_order': False, 'shape': (";
  header += std::to_string(rows);
  header += ", ";
  header += std::to_string(cols);
  header += "), }";

  // 计算 padding： (10 bytes preamble) + (2 bytes header_len) + header + padding 需要 16 对齐
  // 预留换行
  header += '\n';

  // header_len 用 uint16 little-endian
  // 先算需要 padding 到 16-byte 对齐
  size_t preamble = 6 + 2 + 2; // magic(6) + ver(2) + header_len(2) = 10
  size_t header_len = header.size();
  size_t pad_len = 0;
  while ((preamble + header_len + pad_len) % 16 != 0) pad_len++;
  header.insert(header.end() - 1, pad_len, ' '); // 在 '\n' 前插空格 padding
  header_len = header.size();

  require(header_len <= 0xFFFF, "NPY header too large");

  // 写 header length (uint16 LE)
  uint16_t hlen = static_cast<uint16_t>(header_len);
  out.put(static_cast<char>(hlen & 0xFF));
  out.put(static_cast<char>((hlen >> 8) & 0xFF));

  // 写 header + raw data
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(sizeof(T) * (size_t)rows * (size_t)cols));

  require(out.good(), "Write failed: " + path);
}


// 读 .u8bin 格式：[uint32 n][uint32 d][n*d uint8]，原样返回 uint8_t
inline std::vector<uint8_t> read_u8bin_raw(const std::string& path,
                                            int64_t& out_n, int64_t& out_d)
{
  std::ifstream f(path, std::ios::binary);
  require(f.is_open(), "Cannot open: " + path);

  uint32_t n32, d32;
  f.read(reinterpret_cast<char*>(&n32), 4);
  f.read(reinterpret_cast<char*>(&d32), 4);
  require(f.good(), "read_u8bin_raw: failed to read header");

  out_n = static_cast<int64_t>(n32);
  out_d = static_cast<int64_t>(d32);

  std::vector<uint8_t> raw(static_cast<size_t>(out_n) * static_cast<size_t>(out_d));
  f.read(reinterpret_cast<char*>(raw.data()),
         static_cast<std::streamsize>(raw.size()));
  require(f.good(), "read_u8bin_raw: failed to read data");
  return raw;
}

// 读 .u8bin 格式：[uint32 n][uint32 d][n*d uint8]
// 返回转换成 float 的数据，并通过输出参数返回 n、d
inline std::vector<float> read_u8bin(const std::string& path,
                                     int64_t& out_n, int64_t& out_d)
{
  std::ifstream f(path, std::ios::binary);
  require(f.is_open(), "Cannot open: " + path);

  uint32_t n32, d32;
  f.read(reinterpret_cast<char*>(&n32), 4);
  f.read(reinterpret_cast<char*>(&d32), 4);
  require(f.good(), "read_u8bin: failed to read header");

  out_n = static_cast<int64_t>(n32);
  out_d = static_cast<int64_t>(d32);

  std::vector<uint8_t> raw(static_cast<size_t>(out_n) * static_cast<size_t>(out_d));
  f.read(reinterpret_cast<char*>(raw.data()),
         static_cast<std::streamsize>(raw.size()));
  require(f.good(), "read_u8bin: failed to read data");

  std::vector<float> data(raw.size());
  for (size_t i = 0; i < raw.size(); ++i)
    data[i] = static_cast<float>(raw[i]);

  return data;
}

#include <optional>
#include <raft/core/resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/copy.hpp>
#include <raft/neighbors/nn_descent_types.hpp>
// 保存图：idx.graph() -> neighbors.npy
// graph() 直接返回 host_matrix_view，数据已在 host
template <typename IdxT>
void save_graph_as_npy(raft::resources const& res,
                       raft::neighbors::experimental::nn_descent::index<IdxT>& idx,
                       const std::string& path)
{
  auto g = idx.graph();  // host_matrix_view<IdxT, int64_t, row_major>
  int64_t N = g.extent(0);
  int64_t K = g.extent(1);

  write_npy_2d<IdxT>(path, g.data_handle(), N, K);
}

// 保存距离：idx.distances() -> distances.npy
// distances() 返回 optional<device_matrix_view<float,...>>，数据在 device
template <typename IdxT, typename DistT = float>
void save_distances_as_npy(raft::resources const& res,
                           raft::neighbors::experimental::nn_descent::index<IdxT>& idx,
                           const std::string& path)
{
  if (!idx.distances().has_value()) {
    throw std::runtime_error("idx.distances() is empty. "
                             "Set params.return_distances=true when building.");
  }

  auto d = idx.distances().value();  // device_matrix_view<float, int64_t, row_major>
  int64_t N = d.extent(0);
  int64_t K = d.extent(1);

  auto h_dists = raft::make_host_matrix<DistT, int64_t, raft::row_major>(N, K);
  raft::copy(h_dists.data_handle(), d.data_handle(), (size_t)N * (size_t)K,
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  write_npy_2d<DistT>(path, h_dists.data_handle(), N, K);
}
