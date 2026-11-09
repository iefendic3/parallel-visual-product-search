#include "ann/ivf_flat.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

static ann::Dataset sample() {
    ann::Dataset d;
    d.dimensions = 2;
    d.ids = {"A", "B", "C", "D", "E", "F", "G", "H"};
    d.values = {1,1, 1,2, 2,1, 2,2, 8,8, 8,9, 9,8, 9,9};
    return d;
}

int main() {
    const auto dataset = sample();
    const std::vector<float> query{1.2f, 1.3f};
    const auto exact = ann::exact_topk(dataset, query, 4);
    assert(exact.size() == 4);
    assert(exact[0].row_index == 0);
    assert(exact[1].row_index == 1);

    ann::BuildOptions options;
    options.clusters = 2;
    options.iterations = 30;
    options.train_size = 0;
    auto index = ann::IvfFlatIndex::build(sample(), options);
    assert(index.clusters() == 2);
    assert(index.dataset().size() == 8);
    const auto full = index.search(query, 4, index.clusters());
    assert(full.size() == exact.size());
    for (std::size_t i = 0; i < exact.size(); ++i) {
        assert(full[i].row_index == exact[i].row_index);
        assert(std::abs(full[i].squared_distance - exact[i].squared_distance) < 1e-5f);
    }
    assert(ann::recall_at_k(full, exact, 4) == 1.f);
    const auto approximate = index.search(query, 4, 1);
    assert(!approximate.empty());
    assert(ann::recall_at_k(approximate, exact, 4) >= 0.f);

    const auto path = (std::filesystem::temp_directory_path() / "p24_test_index.ivf").string();
    index.save(path);
    auto restored = ann::IvfFlatIndex::load(path);
    std::filesystem::remove(path);
    const auto again = restored.search(query, 4, 2);
    assert(again.size() == exact.size());
    for (std::size_t i = 0; i < again.size(); ++i) assert(again[i].row_index == exact[i].row_index);

    bool rejected = false;
    try { index.search({1.0f}, 1, 1); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { index.search(query, 3, 0); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    assert(ann::exact_topk(dataset, query, 50).size() == dataset.size());
    std::cout << "All sequential IVF-Flat tests passed\n";
}
