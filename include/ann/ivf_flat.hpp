#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ann {

struct Dataset {
    std::size_t dimensions = 0;
    std::vector<std::string> ids;
    std::vector<float> values; // contiguous row-major, N x dimensions

    std::size_t size() const { return ids.size(); }
    const float* row(std::size_t index) const { return values.data() + index * dimensions; }
    void validate() const;
};

Dataset read_csv(const std::string& filename);
std::vector<float> parse_vector(const std::string& text);

struct Neighbor {
    std::uint32_t row_index;
    float squared_distance;
};

struct BuildOptions {
    std::size_t clusters = 4;
    std::size_t iterations = 20;
    std::size_t train_size = 20000; // cap on sampled training rows; 0 means all
    std::uint64_t seed = 42;
};

float squared_l2(const float* a, const float* b, std::size_t dimensions);
std::vector<Neighbor> exact_topk(const Dataset& dataset, const std::vector<float>& query, std::size_t k);
float recall_at_k(const std::vector<Neighbor>& retrieved, const std::vector<Neighbor>& exact, std::size_t k);

class IvfFlatIndex {
public:
    static IvfFlatIndex build(Dataset dataset, const BuildOptions& options);
    static IvfFlatIndex load(const std::string& filename);
    void save(const std::string& filename) const;

    std::vector<Neighbor> search(const std::vector<float>& query, std::size_t k, std::size_t nprobe) const;
    const Dataset& dataset() const { return dataset_; }
    std::size_t clusters() const { return lists_.size(); }
    const std::vector<std::vector<std::uint32_t>>& lists() const { return lists_; }

private:
    Dataset dataset_;
    std::vector<float> centroids_;
    std::vector<std::vector<std::uint32_t>> lists_;
};

} // namespace ann
