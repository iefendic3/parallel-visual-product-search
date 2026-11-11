#include "ann/ivf_flat.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Args = std::unordered_map<std::string, std::string>;

Args read_args(int argc, char** argv, int first) {
    Args args;
    for (int i = first; i < argc; i += 2) {
        const std::string name(argv[i]);
        if (name.rfind("--", 0) != 0 || i + 1 >= argc)
            throw std::runtime_error("Expected --option value pairs");
        if (!args.emplace(name.substr(2), argv[i + 1]).second)
            throw std::runtime_error("Duplicate option: " + name);
    }
    return args;
}

std::string required(const Args& args, const std::string& key) {
    const auto it = args.find(key);
    if (it == args.end()) throw std::runtime_error("Missing required --" + key);
    return it->second;
}

std::size_t natural(const Args& args, const std::string& key, std::size_t fallback, bool allow_zero = false) {
    const auto it = args.find(key);
    if (it == args.end()) return fallback;
    std::size_t parsed = 0;
    unsigned long long number = 0;
    try { number = std::stoull(it->second, &parsed); }
    catch (...) { throw std::runtime_error("Invalid integer for --" + key); }
    if (parsed != it->second.size() || it->second.empty() || it->second[0] == '-' ||
        (!allow_zero && number == 0) || number > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("Invalid integer for --" + key);
    return static_cast<std::size_t>(number);
}

double positive_seconds(const Args& args, const std::string& key, double fallback) {
    const auto it = args.find(key);
    if (it == args.end()) return fallback;
    std::size_t parsed = 0;
    double seconds = 0.0;
    try { seconds = std::stod(it->second, &parsed); }
    catch (...) { throw std::runtime_error("Invalid seconds for --" + key); }
    if (parsed != it->second.size() || !std::isfinite(seconds) || seconds <= 0.0)
        throw std::runtime_error("--" + key + " must be a positive, finite number of seconds");
    return seconds;
}

// Fixed-cost consumption of the result size and both ends of the ranked result.
// Printing the rolling checksum makes the search outputs observable at -O2.
// memcpy reads float bits without aliasing violations; unsigned overflow is defined.
std::uint64_t consume_results(std::uint64_t checksum, const std::vector<ann::Neighbor>& results) {
    constexpr std::uint64_t multiplier = 1099511628211ULL;
    checksum = (checksum ^ static_cast<std::uint64_t>(results.size())) * multiplier;
    if (!results.empty()) {
        for (const auto* neighbor : {&results.front(), &results.back()}) {
            std::uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(neighbor->squared_distance), "Expected 32-bit floats");
            std::memcpy(&bits, &neighbor->squared_distance, sizeof(bits));
            const auto value = (static_cast<std::uint64_t>(neighbor->row_index) << 32) | bits;
            checksum = (checksum ^ value) * multiplier;
        }
    }
    return checksum;
}

struct BenchResult {
    static constexpr std::size_t sample_capacity = 65536;
    static constexpr std::uint64_t sampling_seed = 42;
    std::uint64_t searches = 0;
    std::uint64_t checksum = 14695981039346656037ULL;
    std::uint64_t warmup_checksum = 14695981039346656037ULL;
    double measured_seconds = 0.0;
    double search_us = 0.0;
    std::vector<double> latencies;
};

template <typename Search>
BenchResult measure_searches(const std::vector<std::vector<float>>& queries, Search search,
                            double min_duration, std::size_t warmup) {
    BenchResult result;
    result.latencies.reserve(BenchResult::sample_capacity);
    std::mt19937_64 sampling_rng(BenchResult::sampling_seed);

    std::size_t query_index = 0;
    for (std::size_t i = 0; i < warmup; ++i) {
        result.warmup_checksum = consume_results(result.warmup_checksum, search(queries[query_index]));
        if (++query_index == queries.size()) query_index = 0;
    }
    // Both modes start measured work at query zero, regardless of the warmup count.
    query_index = 0;
    std::cerr << "Warmup complete; starting measured search phase (minimum " << min_duration << " s)\n";
    const auto measured_start = Clock::now();
    Clock::time_point search_end;
    do {
        double latency_us;
        {
            const auto search_start = Clock::now();
            const auto neighbors = search(queries[query_index]);
            search_end = Clock::now();
            latency_us = std::chrono::duration<double, std::micro>(search_end - search_start).count();
            result.checksum = consume_results(result.checksum, neighbors);
        }
        ++result.searches;
        result.search_us += latency_us;
        // Reservoir sampling: uniformly retain at most 512 KiB of latency data.
        // Latency-buffer writes, result hashing, sorting, and output are outside search timing.
        if (result.latencies.size() < BenchResult::sample_capacity) {
            result.latencies.push_back(latency_us);
        } else {
            const auto slot = std::uniform_int_distribution<std::uint64_t>(0, result.searches - 1)(sampling_rng);
            if (slot < BenchResult::sample_capacity)
                result.latencies[static_cast<std::size_t>(slot)] = latency_us;
        }
        if (++query_index == queries.size()) query_index = 0;
    } while (std::chrono::duration<double>(search_end - measured_start).count() < min_duration);
    result.measured_seconds = std::chrono::duration<double>(Clock::now() - measured_start).count();
    // Statistics are calculated only after the measured phase finishes.
    std::sort(result.latencies.begin(), result.latencies.end());
    return result;
}

void benchmark(const Args& args) {
    const std::unordered_set<std::string> allowed{
        "index", "queries", "mode", "topk", "nprobe", "min-duration", "warmup"};
    for (const auto& arg : args) {
        if (!allowed.count(arg.first)) throw std::runtime_error("Unknown bench option: --" + arg.first);
        if ((arg.first == "topk" || arg.first == "nprobe" || arg.first == "warmup") &&
            (arg.second.empty() || arg.second.find_first_not_of("0123456789") != std::string::npos))
            throw std::runtime_error("Invalid integer for --" + arg.first);
    }
    const auto mode_it = args.find("mode");
    const std::string mode = mode_it == args.end() ? "ann" : mode_it->second;
    if (mode != "ann" && mode != "exact") throw std::runtime_error("--mode must be ann or exact");
    const auto k = natural(args, "topk", 10);
    const auto nprobe = natural(args, "nprobe", 1);
    const auto warmup = natural(args, "warmup", 100, true);
    const auto min_duration = positive_seconds(args, "min-duration", 10.0);
    const auto index_path = required(args, "index");
    const auto queries_path = required(args, "queries");
    const auto index = ann::IvfFlatIndex::load(index_path);
    if (mode == "ann" && nprobe > index.clusters())
        throw std::runtime_error("nprobe must be in [1, nlist]");

    // Materialize each query once so the timed loop does not copy 512D vectors.
    const auto queries = [&] {
        const auto data = ann::read_csv(queries_path);
        if (data.dimensions != index.dataset().dimensions)
            throw std::runtime_error("Query and index dimensions differ");
        std::vector<std::vector<float>> vectors;
        vectors.reserve(data.size());
        for (std::size_t i = 0; i < data.size(); ++i)
            vectors.emplace_back(data.row(i), data.row(i) + data.dimensions);
        return vectors;
    }();

    // Dispatch outside the hot loop. Only the chosen algorithm runs, including warmup.
    const auto result = mode == "ann"
        ? measure_searches(queries, [&](const auto& q) { return index.search(q, k, nprobe); }, min_duration, warmup)
        : measure_searches(queries, [&](const auto& q) { return ann::exact_topk(index.dataset(), q, k); }, min_duration, warmup);
    const auto middle = result.latencies.size() / 2;
    const auto median = result.latencies.size() % 2
        ? result.latencies[middle]
        : (result.latencies[middle - 1] + result.latencies[middle]) * 0.5;
    const auto p95_rank = static_cast<std::size_t>(std::ceil(0.95 * result.latencies.size()));
    std::cout << "mode=" << mode << '\n'
              << "index=" << index_path << '\n'
              << "queries=" << queries_path << '\n'
              << "index_vectors=" << index.dataset().size() << '\n'
              << "dimensions=" << index.dataset().dimensions << '\n'
              << "clusters=" << index.clusters() << '\n'
              << "query_count=" << queries.size() << '\n'
              << "topk=" << k << '\n'
              << "nprobe=" << (mode == "ann" ? std::to_string(nprobe) : "not_applicable") << '\n'
              << "warmup=" << warmup << '\n'
              << std::fixed << std::setprecision(9)
              << "min_duration_seconds=" << min_duration << '\n'
              << "measured_seconds=" << result.measured_seconds << '\n'
              << "timed_search_seconds=" << result.search_us / 1e6 << '\n'
              << "searches=" << result.searches << '\n'
              << std::setprecision(4)
              << "mean_search_us=" << result.search_us / result.searches << '\n'
              << "median_search_us=" << median << '\n'
              << "p95_search_us=" << result.latencies[p95_rank - 1] << '\n'
              << "queries_per_second=" << result.searches / result.measured_seconds << '\n'
              << "latency_samples=" << result.latencies.size() << '\n'
              << "latency_sample_capacity=" << BenchResult::sample_capacity << '\n'
              << "latency_sampling=" << (result.searches <= BenchResult::sample_capacity ? "all" : "reservoir") << '\n'
              << "latency_sampling_seed=" << BenchResult::sampling_seed << '\n'
              << "checksum=" << result.checksum << '\n'
              << "warmup_checksum=" << result.warmup_checksum << '\n';
}

void usage() {
    std::cout << "P24 sequential IVF-Flat ANN (squared L2)\n"
              << "  ann_cli build --input vectors.csv --output index.ivf --clusters 4 [--iterations 20] [--train-size 20000] [--seed 42]\n"
              << "  ann_cli search --index index.ivf --query 0.1,0.2 --topk 3 --nprobe 2\n"
              << "  ann_cli evaluate --index index.ivf --queries queries.csv --topk 3 --nprobe 2 [--limit 100]\n"
              << "  ann_cli bench --index index.ivf --queries queries.csv [--mode ann|exact] [--topk 10] [--nprobe 1] [--min-duration 10] [--warmup 100]\n"
              << "CSV: one ID followed by numeric coordinates per row; no header.\n";
}

void print_neighbors(const ann::Dataset& data, const std::vector<ann::Neighbor>& results) {
    std::cout << "rank,id,squared_l2,l2\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << (i + 1) << ',' << data.ids[r.row_index] << ','
                  << std::setprecision(9) << r.squared_distance << ','
                  << std::sqrt(r.squared_distance) << '\n';
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "help") {
            usage();
            return argc < 2 ? 1 : 0;
        }
        const std::string command(argv[1]);
        const auto args = read_args(argc, argv, 2);
        if (command == "bench") {
            benchmark(args);
            return 0;
        }
        if (command == "build") {
            auto data = ann::read_csv(required(args, "input"));
            ann::BuildOptions options;
            options.clusters = natural(args, "clusters", 4);
            options.iterations = natural(args, "iterations", 20);
            options.train_size = natural(args, "train-size", 20000, true);
            options.seed = natural(args, "seed", 42, true);
            const auto start = Clock::now();
            auto index = ann::IvfFlatIndex::build(std::move(data), options);
            const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
            index.save(required(args, "output"));
            std::cerr << "Built IVF-Flat index: vectors=" << index.dataset().size()
                      << " dimensions=" << index.dataset().dimensions
                      << " clusters=" << index.clusters()
                      << " train_seconds=" << seconds << '\n';
            return 0;
        }
        if (command == "search") {
            auto index = ann::IvfFlatIndex::load(required(args, "index"));
            auto query = ann::parse_vector(required(args, "query"));
            const auto k = natural(args, "topk", 10);
            const auto nprobe = natural(args, "nprobe", 1);
            const auto start = Clock::now();
            auto result = index.search(query, k, nprobe);
            const double microseconds = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
            print_neighbors(index.dataset(), result);
            std::cerr << "search_us=" << microseconds << " results=" << result.size() << '\n';
            return 0;
        }
        if (command == "evaluate") {
            auto index = ann::IvfFlatIndex::load(required(args, "index"));
            auto queries = ann::read_csv(required(args, "queries"));
            if (queries.dimensions != index.dataset().dimensions)
                throw std::runtime_error("Query and index dimensions differ");
            const auto k = natural(args, "topk", 10);
            const auto nprobe = natural(args, "nprobe", 1);
            const auto limit = std::min(natural(args, "limit", queries.size()), queries.size());
            double recall = 0.0;
            double ann_us = 0.0;
            double exact_us = 0.0;
            for (std::size_t i = 0; i < limit; ++i) {
                std::vector<float> query(queries.row(i), queries.row(i) + queries.dimensions);
                auto start = Clock::now();
                const auto approximate = index.search(query, k, nprobe);
                ann_us += std::chrono::duration<double, std::micro>(Clock::now() - start).count();
                start = Clock::now();
                const auto exact = ann::exact_topk(index.dataset(), query, k);
                exact_us += std::chrono::duration<double, std::micro>(Clock::now() - start).count();
                recall += ann::recall_at_k(approximate, exact, k);
            }
            std::cout << std::fixed << std::setprecision(4)
                      << "queries=" << limit << '\n'
                      << "recall_at_" << k << '=' << recall / limit << '\n'
                      << "ann_mean_us=" << ann_us / limit << '\n'
                      << "exact_mean_us=" << exact_us / limit << '\n'
                      << "search_speedup_vs_exact=" << (ann_us > 0.0 ? exact_us / ann_us : 0.0) << '\n';
            return 0;
        }
        usage();
        throw std::runtime_error("Unknown command: " + command);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
