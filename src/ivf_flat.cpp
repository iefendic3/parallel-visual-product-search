#include "ann/ivf_flat.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace ann {
namespace {

std::string trim(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

float parse_float(const std::string& token) {
    const std::string clean = trim(token);
    std::size_t used = 0;
    float value;
    try {
        value = std::stof(clean, &used);
    } catch (...) {
        throw std::runtime_error("Invalid float value: " + clean);
    }
    if (used != clean.size() || !std::isfinite(value))
        throw std::runtime_error("Invalid or non-finite float value: " + clean);
    return value;
}

bool better(const Neighbor& a, const Neighbor& b) {
    if (a.squared_distance != b.squared_distance) return a.squared_distance < b.squared_distance;
    return a.row_index < b.row_index;
}

// priority_queue's top is the worst of the k retained candidates.
struct BetterComparator {
    bool operator()(const Neighbor& a, const Neighbor& b) const { return better(a, b); }
};

class TopK {
public:
    explicit TopK(std::size_t k) : k_(k) {}
    void add(Neighbor candidate) {
        if (k_ == 0) return;
        if (heap_.size() < k_) heap_.push(candidate);
        else if (better(candidate, heap_.top())) {
            heap_.pop();
            heap_.push(candidate);
        }
    }
    std::vector<Neighbor> finish() {
        std::vector<Neighbor> results;
        while (!heap_.empty()) {
            results.push_back(heap_.top());
            heap_.pop();
        }
        std::sort(results.begin(), results.end(), better);
        return results;
    }
private:
    std::size_t k_;
    std::priority_queue<Neighbor, std::vector<Neighbor>, BetterComparator> heap_;
};

std::size_t nearest_centroid(const float* vector, const std::vector<float>& centroids,
                             std::size_t nlist, std::size_t dims, float* best_d2 = nullptr) {
    float smallest = std::numeric_limits<float>::infinity();
    std::size_t closest = 0;
    for (std::size_t c = 0; c < nlist; ++c) {
        const float d = squared_l2(vector, centroids.data() + c * dims, dims);
        if (d < smallest) { smallest = d; closest = c; }
    }
    if (best_d2) *best_d2 = smallest;
    return closest;
}

template <typename T>
void write_scalar(std::ofstream& file, T value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!file) throw std::runtime_error("Failed writing index file");
}

template <typename T>
T read_scalar(std::ifstream& file) {
    T value{};
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!file) throw std::runtime_error("Incomplete or corrupted index file");
    return value;
}

void write_floats(std::ofstream& file, const std::vector<float>& v) {
    file.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    if (!file) throw std::runtime_error("Failed writing vectors");
}

void read_floats(std::ifstream& file, std::vector<float>& v) {
    file.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    if (!file) throw std::runtime_error("Incomplete vector contents in index file");
    for (const auto x : v) if (!std::isfinite(x)) throw std::runtime_error("Non-finite data in index file");
}

} // namespace

void Dataset::validate() const {
    if (dimensions == 0 || ids.empty()) throw std::runtime_error("Dataset must have at least one row and dimension");
    if (ids.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("This reference implementation supports at most 2^32-1 rows");
    if (ids.size() > std::numeric_limits<std::size_t>::max() / dimensions ||
        values.size() != ids.size() * dimensions)
        throw std::runtime_error("Dataset dimensions do not match storage");
    for (const auto v : values) if (!std::isfinite(v)) throw std::runtime_error("Non-finite vector");
}

std::vector<float> parse_vector(const std::string& text) {
    std::vector<float> values;
    std::stringstream input(text);
    std::string part;
    while (std::getline(input, part, ',')) values.push_back(parse_float(part));
    if (values.empty()) throw std::runtime_error("Query must contain coordinates separated by commas");
    return values;
}

Dataset read_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open CSV: " + filename);
    Dataset out;
    std::unordered_set<std::string> unique_ids;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const auto comma = line.find(',');
        if (comma == std::string::npos) throw std::runtime_error("Expected id,v1,... on CSV line " + std::to_string(line_number));
        const std::string id = trim(line.substr(0, comma));
        if (id.empty()) throw std::runtime_error("Empty ID on line " + std::to_string(line_number));
        if (!unique_ids.insert(id).second) throw std::runtime_error("Duplicate ID: " + id);
        const auto values = parse_vector(line.substr(comma + 1));
        if (out.dimensions == 0) out.dimensions = values.size();
        if (values.size() != out.dimensions)
            throw std::runtime_error("Inconsistent dimension on line " + std::to_string(line_number));
        out.ids.push_back(id);
        out.values.insert(out.values.end(), values.begin(), values.end());
    }
    out.validate();
    return out;
}

float squared_l2(const float* a, const float* b, std::size_t dimensions) {
    float d2 = 0.f;
    for (std::size_t i = 0; i < dimensions; ++i) {
        const float delta = a[i] - b[i];
        d2 += delta * delta;
    }
    return d2;
}

std::vector<Neighbor> exact_topk(const Dataset& dataset, const std::vector<float>& query, std::size_t k) {
    // Data are validated at load/build time. Do not rescan every vector on every query:
    // that would artificially double the exact baseline's memory traffic.
    if (dataset.dimensions == 0 || dataset.size() == 0 ||
        dataset.size() > std::numeric_limits<std::size_t>::max() / dataset.dimensions ||
        dataset.values.size() != dataset.size() * dataset.dimensions)
        throw std::runtime_error("Invalid dataset storage");
    if (query.size() != dataset.dimensions) throw std::runtime_error("Query dimensionality mismatch");
    if (k == 0) throw std::runtime_error("topk must be positive");
    TopK top(k);
    for (std::size_t i = 0; i < dataset.size(); ++i)
        top.add({static_cast<std::uint32_t>(i), squared_l2(query.data(), dataset.row(i), dataset.dimensions)});
    return top.finish();
}

float recall_at_k(const std::vector<Neighbor>& retrieved, const std::vector<Neighbor>& exact, std::size_t k) {
    if (k == 0 || exact.empty()) throw std::runtime_error("Recall requires k > 0 and an exact result set");
    std::unordered_set<std::uint32_t> correct;
    for (std::size_t i = 0; i < std::min(k, exact.size()); ++i) correct.insert(exact[i].row_index);
    std::size_t matches = 0;
    for (std::size_t i = 0; i < std::min(k, retrieved.size()); ++i)
        if (correct.count(retrieved[i].row_index)) ++matches;
    return static_cast<float>(matches) / static_cast<float>(correct.size());
}

IvfFlatIndex IvfFlatIndex::build(Dataset dataset, const BuildOptions& options) {
    dataset.validate();
    if (options.clusters == 0 || options.clusters > dataset.size())
        throw std::runtime_error("clusters must be between 1 and dataset size");
    if (options.iterations == 0) throw std::runtime_error("iterations must be positive");
    const std::size_t dims = dataset.dimensions;
    const std::size_t nlist = options.clusters;
    const std::size_t sample_size = options.train_size == 0 ? dataset.size() : std::min(dataset.size(), options.train_size);
    if (sample_size < nlist) throw std::runtime_error("train_size must be >= clusters");

    std::mt19937_64 rng(options.seed);
    std::vector<std::uint32_t> training(dataset.size());
    std::iota(training.begin(), training.end(), 0u);
    std::shuffle(training.begin(), training.end(), rng);
    training.resize(sample_size);

    std::vector<float> centroids(nlist * dims);
    std::uniform_int_distribution<std::size_t> choose_sample(0, sample_size - 1);
    auto set_center = [&](std::size_t c, std::size_t training_slot) {
        const float* source = dataset.row(training[training_slot]);
        std::copy(source, source + dims, centroids.begin() + static_cast<std::ptrdiff_t>(c * dims));
    };
    set_center(0, choose_sample(rng));
    // k-means++ initialization: sample subsequent centers with probability proportional to squared distance.
    std::vector<double> min_d2(sample_size, std::numeric_limits<double>::infinity());
    for (std::size_t c = 1; c < nlist; ++c) {
        double total = 0.0;
        for (std::size_t i = 0; i < sample_size; ++i) {
            const auto d = squared_l2(dataset.row(training[i]), centroids.data() + (c - 1) * dims, dims);
            min_d2[i] = std::min(min_d2[i], static_cast<double>(d));
            total += min_d2[i];
        }
        if (total <= 0.0) {
            set_center(c, choose_sample(rng));
        } else {
            std::uniform_real_distribution<double> sample_weight(0.0, total);
            const double choice = sample_weight(rng);
            double cumulative = 0.0;
            std::size_t chosen = sample_size - 1;
            for (std::size_t i = 0; i < sample_size; ++i) {
                cumulative += min_d2[i];
                if (cumulative >= choice) { chosen = i; break; }
            }
            set_center(c, chosen);
        }
    }

    std::vector<std::size_t> assignments(sample_size, nlist);
    for (std::size_t iteration = 0; iteration < options.iterations; ++iteration) {
        bool any_changed = false;
        std::vector<double> sums(nlist * dims, 0.0);
        std::vector<std::size_t> counts(nlist, 0);
        for (std::size_t i = 0; i < sample_size; ++i) {
            const auto c = nearest_centroid(dataset.row(training[i]), centroids, nlist, dims);
            if (c != assignments[i]) any_changed = true;
            assignments[i] = c;
            ++counts[c];
            const float* v = dataset.row(training[i]);
            for (std::size_t d = 0; d < dims; ++d) sums[c * dims + d] += v[d];
        }
        for (std::size_t c = 0; c < nlist; ++c) {
            if (counts[c] == 0) {
                set_center(c, choose_sample(rng));
                any_changed = true;
            } else {
                for (std::size_t d = 0; d < dims; ++d)
                    centroids[c * dims + d] = static_cast<float>(sums[c * dims + d] / counts[c]);
            }
        }
        if (!any_changed) break;
    }

    IvfFlatIndex index;
    index.dataset_ = std::move(dataset);
    index.centroids_ = std::move(centroids);
    index.lists_.resize(nlist);
    // Final assignment indexes the entire dataset, even if k-means was trained on a subset.
    for (std::size_t i = 0; i < index.dataset_.size(); ++i) {
        const auto c = nearest_centroid(index.dataset_.row(i), index.centroids_, nlist, dims);
        index.lists_[c].push_back(static_cast<std::uint32_t>(i));
    }
    return index;
}

std::vector<Neighbor> IvfFlatIndex::search(const std::vector<float>& query, std::size_t k, std::size_t nprobe) const {
    if (query.size() != dataset_.dimensions) throw std::runtime_error("Query dimensionality mismatch");
    if (k == 0) throw std::runtime_error("topk must be positive");
    if (nprobe == 0 || nprobe > lists_.size()) throw std::runtime_error("nprobe must be in [1, nlist]");

    std::vector<std::pair<float, std::size_t>> by_centroid;
    by_centroid.reserve(lists_.size());
    for (std::size_t c = 0; c < lists_.size(); ++c)
        by_centroid.emplace_back(squared_l2(query.data(), centroids_.data() + c * dataset_.dimensions,
                                             dataset_.dimensions), c);
    std::partial_sort(by_centroid.begin(), by_centroid.begin() + static_cast<std::ptrdiff_t>(nprobe),
                      by_centroid.end());

    TopK top(k);
    for (std::size_t probe = 0; probe < nprobe; ++probe) {
        for (std::uint32_t row : lists_[by_centroid[probe].second])
            top.add({row, squared_l2(query.data(), dataset_.row(row), dataset_.dimensions)});
    }
    return top.finish();
}

void IvfFlatIndex::save(const std::string& filename) const {
    dataset_.validate();
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot write index: " + filename);
    file.write("P24IVF01", 8);
    write_scalar<std::uint32_t>(file, 1);
    write_scalar<std::uint32_t>(file, static_cast<std::uint32_t>(dataset_.dimensions));
    write_scalar<std::uint64_t>(file, static_cast<std::uint64_t>(dataset_.size()));
    write_scalar<std::uint32_t>(file, static_cast<std::uint32_t>(lists_.size()));
    for (const auto& id : dataset_.ids) {
        if (id.size() > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error("ID too long");
        write_scalar<std::uint32_t>(file, static_cast<std::uint32_t>(id.size()));
        file.write(id.data(), static_cast<std::streamsize>(id.size()));
    }
    write_floats(file, dataset_.values);
    write_floats(file, centroids_);
    for (const auto& list : lists_) {
        write_scalar<std::uint64_t>(file, static_cast<std::uint64_t>(list.size()));
        for (std::uint32_t row : list) write_scalar(file, row);
    }
    if (!file) throw std::runtime_error("Index save failed");
}

IvfFlatIndex IvfFlatIndex::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open index: " + filename);
    char magic[8];
    file.read(magic, 8);
    if (!file || std::string(magic, 8) != "P24IVF01") throw std::runtime_error("Not a P24 IVF index");
    if (read_scalar<std::uint32_t>(file) != 1) throw std::runtime_error("Unsupported index version");
    const auto dims = read_scalar<std::uint32_t>(file);
    const auto count = read_scalar<std::uint64_t>(file);
    const auto nlist = read_scalar<std::uint32_t>(file);
    if (dims == 0 || count == 0 || count > std::numeric_limits<std::uint32_t>::max() ||
        nlist == 0 || nlist > count || count > std::numeric_limits<std::size_t>::max() / dims)
        throw std::runtime_error("Invalid index dimensions or cluster count");
    IvfFlatIndex result;
    result.dataset_.dimensions = dims;
    result.dataset_.ids.reserve(static_cast<std::size_t>(count));
    for (std::size_t i = 0; i < count; ++i) {
        const auto length = read_scalar<std::uint32_t>(file);
        if (length == 0 || length > 1'000'000) throw std::runtime_error("Invalid index ID length");
        std::string id(length, '\0');
        file.read(id.data(), length);
        if (!file) throw std::runtime_error("Incomplete index ID");
        result.dataset_.ids.push_back(std::move(id));
    }
    result.dataset_.values.resize(static_cast<std::size_t>(count) * dims);
    read_floats(file, result.dataset_.values);
    result.centroids_.resize(static_cast<std::size_t>(nlist) * dims);
    read_floats(file, result.centroids_);
    result.lists_.resize(nlist);
    std::vector<bool> visited(static_cast<std::size_t>(count), false);
    std::size_t assigned = 0;
    for (auto& list : result.lists_) {
        const auto length = read_scalar<std::uint64_t>(file);
        if (length > count || assigned + length > count) throw std::runtime_error("Invalid inverted list size");
        list.reserve(static_cast<std::size_t>(length));
        for (std::size_t j = 0; j < length; ++j) {
            const auto row = read_scalar<std::uint32_t>(file);
            if (row >= count || visited[row]) throw std::runtime_error("Invalid or duplicate row in inverted lists");
            visited[row] = true;
            list.push_back(row);
        }
        assigned += static_cast<std::size_t>(length);
    }
    if (assigned != count) throw std::runtime_error("Missing vector assignments in index");
    result.dataset_.validate();
    return result;
}

} // namespace ann
