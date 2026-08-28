// load.hpp
// C++ helpers to read BIGANN-style data files (.fbin/.bin/.u8bin/.i8bin/.ibin)
// and write index in the SAME format as Python's: np.save("neighbors.npy", int64[N,M])
//
// Python format you used:
//   neighbors: shape (N, M), dtype int64, padding -1
//   np.save(path, neighbors)
//
// This file provides:
//   - read_fbin_f32 / read_u8bin_to_f32 / read_ibin_i32
//   - write_npy_int64_2d(path, data, N, M)    // writes .npy v1.0, little-endian int64
//
// Build note: header-only; just include in your .cpp and compile with -std=c++17.

#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

struct Neighbor {
    int id;
    float dist;
    bool is_new;
};

namespace load {

// ---------- small utils ----------
inline void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

inline std::pair<int32_t,int32_t> read_bigann_header(std::ifstream& in) {
    int32_t N=0, D=0;
    in.read(reinterpret_cast<char*>(&N), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&D), sizeof(int32_t));
    require(in.good(), "Failed to read BIGANN header (N,D).");
    require(N > 0 && D > 0, "Invalid BIGANN header (N<=0 or D<=0).");
    return {N, D};
}

// ---------- data readers ----------
// .fbin / .bin: header int32 N,D then N*D float32
inline void read_fbin_f32(const std::string& path,
                          std::vector<float>& out,
                          int32_t& N, int32_t& D) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "Cannot open file: " + path);

    auto [n, d] = read_bigann_header(in);
    N = n; D = d;

    const uint64_t cnt = static_cast<uint64_t>(N) * static_cast<uint64_t>(D);
    out.resize(cnt);

    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(cnt * sizeof(float)));
    require(in.good(), "Failed to read float payload from: " + path);
}

// .u8bin / .i8bin: header int32 N,D then N*D uint8, convert to float32
inline void read_u8bin_to_f32(const std::string& path,
                              std::vector<float>& out,
                              int32_t& N, int32_t& D) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "Cannot open file: " + path);

    auto [n, d] = read_bigann_header(in);
    N = n; D = d;

    const uint64_t cnt = static_cast<uint64_t>(N) * static_cast<uint64_t>(D);

    std::vector<uint8_t> tmp(cnt);
    in.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(cnt));
    require(in.good(), "Failed to read uint8 payload from: " + path);

    out.resize(cnt);
    for (uint64_t i = 0; i < cnt; ++i) out[i] = static_cast<float>(tmp[i]);
}

// .ibin: header int32 N,D then N*D int32
inline void read_ibin_i32(const std::string& path,
                          std::vector<int32_t>& out,
                          int32_t& N, int32_t& D) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "Cannot open file: " + path);

    auto [n, d] = read_bigann_header(in);
    N = n; D = d;

    const uint64_t cnt = static_cast<uint64_t>(N) * static_cast<uint64_t>(D);
    out.resize(cnt);

    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(cnt * sizeof(int32_t)));
    require(in.good(), "Failed to read int32 payload from: " + path);
}

// ---------- NPY reader for int64 2D array (matches np.save / np.load) ----------
// Reads .npy v1.0/v2.0, little-endian int64, C-order, shape (N,M).
inline void read_npy_int64_2d(const std::string& path,
                               std::vector<int64_t>& out,
                               int64_t& N, int64_t& M) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "Cannot open file: " + path);

    // magic: \x93NUMPY
    char magic[6];
    in.read(magic, 6);
    require(in.good() && magic[0] == '\x93' && std::string(magic+1, 5) == "NUMPY",
            "Not a valid .npy file: " + path);

    // version
    uint8_t ver_major, ver_minor;
    in.read(reinterpret_cast<char*>(&ver_major), 1);
    in.read(reinterpret_cast<char*>(&ver_minor), 1);

    // header length
    uint32_t header_len = 0;
    if (ver_major == 1) {
        uint16_t hlen16;
        in.read(reinterpret_cast<char*>(&hlen16), sizeof(uint16_t));
        header_len = hlen16;
    } else if (ver_major == 2) {
        in.read(reinterpret_cast<char*>(&header_len), sizeof(uint32_t));
    } else {
        throw std::runtime_error("Unsupported .npy version: " + std::to_string(ver_major));
    }

    // read header string
    std::string header(header_len, '\0');
    in.read(&header[0], header_len);
    require(in.good(), "Failed to read .npy header from: " + path);

    // parse shape from header: 'shape': (N, M)
    auto shape_pos = header.find("'shape'");
    require(shape_pos != std::string::npos, "No 'shape' in .npy header: " + path);
    auto paren_open = header.find('(', shape_pos);
    auto paren_close = header.find(')', paren_open);
    require(paren_open != std::string::npos && paren_close != std::string::npos,
            "Malformed shape in .npy header: " + path);
    std::string shape_str = header.substr(paren_open + 1, paren_close - paren_open - 1);

    // parse two integers from "N, M"
    auto comma = shape_str.find(',');
    require(comma != std::string::npos, "Expected 2D shape in .npy: " + path);
    N = std::stoll(shape_str.substr(0, comma));
    M = std::stoll(shape_str.substr(comma + 1));
    require(N > 0 && M > 0, "Invalid shape in .npy: " + path);

    // read payload
    uint64_t cnt = static_cast<uint64_t>(N) * static_cast<uint64_t>(M);
    out.resize(cnt);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(cnt * sizeof(int64_t)));
    require(in.good(), "Failed to read int64 payload from: " + path);
}

// ---------- NPY writer (matches np.save for a contiguous int64 2D array) ----------
// Writes .npy v1.0, little-endian int64, C-order, shape (N,M).
//
// Equivalent in Python:
//   np.save(path, arr.astype(np.int64, copy=False))
inline void write_npy_int64_2d(const std::string& path,
                               const int64_t* data,
                               int64_t N, int64_t M) {
    require(N > 0 && M > 0, "write_npy_int64_2d: N and M must be > 0");
    std::ofstream out(path, std::ios::binary);
    require(out.is_open(), "Cannot open for write: " + path);

    // magic + version
    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);
    const uint8_t ver_major = 1;
    const uint8_t ver_minor = 0;
    out.put(static_cast<char>(ver_major));
    out.put(static_cast<char>(ver_minor));

    // build header dict (must end with newline, and header padding to 16-byte alignment of preamble+header)
    // dtype: little-endian int64 = "<i8"
    // fortran_order: False
    // shape: (N, M)
    std::string header = "{'descr': '<i8', 'fortran_order': False, 'shape': (";
    header += std::to_string(N);
    header += ", ";
    header += std::to_string(M);
    header += "), }";

    // header must be padded with spaces to reach alignment, and end with '\n'
    // For v1.0: header length is stored as uint16 little-endian
    // Total: magic(6)+ver(2)+hlen(2)+header = preamble(10)+header
    const size_t preamble = 10;
    // +1 for '\n'
    size_t header_len = header.size() + 1;
    // pad so that (preamble + header_len) % 16 == 0
    size_t pad = (16 - ((preamble + header_len) % 16)) % 16;
    header.append(pad, ' ');
    header.push_back('\n');
    header_len = header.size();

    require(header_len <= 65535, "NPY v1.0 header too large.");

    // write header length (uint16 LE)
    uint16_t hlen = static_cast<uint16_t>(header_len);
    out.write(reinterpret_cast<const char*>(&hlen), sizeof(uint16_t));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    // write data payload (C-order contiguous)
    const uint64_t cnt = static_cast<uint64_t>(N) * static_cast<uint64_t>(M);
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(cnt * sizeof(int64_t)));
    require(out.good(), "Failed while writing npy payload.");
}

// Convenience overload for std::vector<int64_t> storing row-major (N*M)
inline void write_npy_int64_2d(const std::string& path,
                               const std::vector<int64_t>& data,
                               int64_t N, int64_t M) {
    require(static_cast<int64_t>(data.size()) == N * M, "Data size != N*M");
    write_npy_int64_2d(path, data.data(), N, M);
}

void save_neighbors_npy(const std::string& path,
                        const std::vector<std::vector<Neighbor>>& graph,
                        int K) {
    int64_t N = static_cast<int64_t>(graph.size());
    int64_t M = static_cast<int64_t>(K);

    std::vector<int64_t> buf(N * M, -1);  // 默认填 -1

    for (int64_t i = 0; i < N; ++i) {
        const auto& neighs = graph[i];

        // 拷贝一份排序（防止原 graph 顺序被改）
        std::vector<Neighbor> tmp = neighs;
        std::sort(tmp.begin(), tmp.end(),
                  [](const Neighbor& a, const Neighbor& b) {
                      return a.dist < b.dist;
                  });

        int limit = std::min<int>(K, tmp.size());
        for (int j = 0; j < limit; ++j) {
            buf[i * M + j] = static_cast<int64_t>(tmp[j].id);
        }
    }

    write_npy_int64_2d(path, buf.data(), N, M);
}


inline void write_npy_f32_2d(const std::string& path,
                             const float* data,
                             int64_t N, int64_t M) {
    require(N > 0 && M > 0, "write_npy_f32_2d: N and M must be > 0");
    std::ofstream out(path, std::ios::binary);
    require(out.is_open(), "Cannot open for write: " + path);

    // magic + version
    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);
    const uint8_t ver_major = 1;
    const uint8_t ver_minor = 0;
    out.put(static_cast<char>(ver_major));
    out.put(static_cast<char>(ver_minor));

    // dtype: little-endian float32 = "<f4"
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    header += std::to_string(N);
    header += ", ";
    header += std::to_string(M);
    header += "), }";

    const size_t preamble = 10;
    size_t header_len = header.size() + 1;
    size_t pad = (16 - ((preamble + header_len) % 16)) % 16;
    header.append(pad, ' ');
    header.push_back('\n');
    header_len = header.size();

    require(header_len <= 65535, "NPY v1.0 header too large.");

    uint16_t hlen = static_cast<uint16_t>(header_len);
    out.write(reinterpret_cast<const char*>(&hlen), sizeof(uint16_t));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));

    const uint64_t cnt = static_cast<uint64_t>(N) * static_cast<uint64_t>(M);
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(cnt * sizeof(float)));

    require(out.good(), "Failed while writing npy payload.");
}

void save_distances_npy(const std::string& path,
                        const std::vector<std::vector<Neighbor>>& graph,
                        int K) {
    int64_t N = static_cast<int64_t>(graph.size());
    int64_t M = static_cast<int64_t>(K);

    std::vector<float> buf(N * M, std::numeric_limits<float>::infinity());

    for (int64_t i = 0; i < N; ++i) {
        std::vector<Neighbor> tmp = graph[i];
        std::sort(tmp.begin(), tmp.end(),
                  [](const Neighbor& a, const Neighbor& b) {
                      return a.dist < b.dist;
                  });

        int limit = std::min<int>(K, tmp.size());
        for (int j = 0; j < limit; ++j) {
            buf[i * M + j] = tmp[j].dist;
        }
    }

    write_npy_f32_2d(path, buf.data(), N, M);
}


} // namespace load
