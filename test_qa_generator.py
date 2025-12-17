"""
Test QA Generator with real data

Usage:
- Uncomment the generator you want to test
- Comment out others
- Run: python test_qa_generator.py

Requirements:
- pip install numpy tqdm (if not installed)
"""

from point_qa_generator import PointQAGenerator, TaskPlan

# Initialize generator
print("🔧 Initializing QA Generator...")
generator = PointQAGenerator(
    metadata_file="./data/test/test_texverse_metadata.jsonl",
    pcd_dir="./data/test/test_npy",
    layouts_file="./data/layout/outputs_gpt_oss/layouts.json",
    seed=42
)
print(f"✅ Loaded {len(generator.metadata.objects)} objects")
print(f"✅ Loaded {len(generator.layouts)} layouts")
print(f"✅ Classified layouts: special={len(generator.layouts_classified['special'])}, standard={len(generator.layouts_classified['standard'])}")
print()

# =============================================================================
# Test Distance Generators
# =============================================================================

# # Test 1: what_distance (closest)
# print("="*70)
# print("Testing: what_distance (closest)")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="what_distance",
#     num_options=4,
#     seed=42,
#     generator_config={"distance_type": "closest"}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/what_distance_closest")
# print()

# # Test 2: what_distance (farthest)
# print("="*70)
# print("Testing: what_distance (farthest)")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="what_distance",
#     num_options=4,
#     seed=42,
#     generator_config={"distance_type": "farthest"}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/what_distance_farthest")
# print()

# Test 3: where_distance (closest)
print("="*70)
print("Testing: where_distance (closest)")
print("="*70)
task_plan = TaskPlan(
    generator_type="where_distance",
    num_options=4,
    seed=42,
    generator_config={"distance_type": "closest"}
)
generator.generate(task_plan, num_tasks=5, output_dir="./output_test/where_distance_closest")
print()

# # Test 4: list_attribute_distance (closest)
# print("="*70)
# print("Testing: list_attribute_distance (closest)")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="list_attribute_distance",
#     num_options=4,
#     seed=42,
#     generator_config={"distance_type": "closest"}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/list_attribute_distance")
# print()

# # Test 5: count_attribute_distance (farthest)
# print("="*70)
# print("Testing: count_attribute_distance (farthest)")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="count_attribute_distance",
#     num_options=4,
#     seed=42,
#     generator_config={"distance_type": "farthest"}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/count_attribute_distance")
# print()


# =============================================================================
# Test Size Generators
# =============================================================================

# # Test 6: what_size (largest)
# print("="*70)
# print("Testing: what_size (largest)")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="what_size",
#     num_options=4,
#     seed=42,
#     generator_config={"size_type": "largest"}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/what_size_largest")
# print()

# # Test 7: what_size (smallest)
# print("="*70)
# print("Testing: what_size (smallest)")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="what_size",
#     num_options=4,
#     seed=42,
#     generator_config={"size_type": "smallest"}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/what_size_smallest")
# print()

# # Test 8: where_size (largest)
# print("="*70)
# print("Testing: where_size (largest)")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="where_size",
#     num_options=4,
#     seed=42,
#     generator_config={"size_type": "largest", "reference_mode": "with_reference"}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/where_size_largest")
# print()

# # Test 9: list_attribute_size (smallest)
# print("="*70)
# print("Testing: list_attribute_size (smallest)")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="list_attribute_size",
#     num_options=4,
#     seed=42,
#     generator_config={"size_type": "smallest"}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/list_attribute_size")
# print()

# # Test 10: count_attribute_size (largest)
# print("="*70)
# print("Testing: count_attribute_size (largest)")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="count_attribute_size",
#     num_options=4,
#     seed=42,
#     generator_config={"size_type": "largest"}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/count_attribute_size")
# print()


# =============================================================================
# Test Attribute Generators
# =============================================================================

# # Test 11: what_attribute
# print("="*70)
# print("Testing: what_attribute")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="what_attribute",
#     num_options=4,
#     seed=42,
#     generator_config={}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/what_attribute")
# print()

# # Test 12: list_attribute
# print("="*70)
# print("Testing: list_attribute")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="list_attribute",
#     num_options=4,
#     seed=42,
#     generator_config={}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/list_attribute")
# print()

# # Test 13: count_attribute
# print("="*70)
# print("Testing: count_attribute")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="count_attribute",
#     num_options=4,
#     seed=42,
#     generator_config={}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/count_attribute")
# print()


# =============================================================================
# Test Number Generators (使用新的Layout驱动方式)
# =============================================================================

# # Test 14: count_object
# print("="*70)
# print("Testing: count_object")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="count_object",
#     num_options=4,
#     seed=42,
#     generator_config={}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/count_object")
# print()

# # Test 15: frequent_object (most)
# print("="*70)
# print("Testing: frequent_object (most)")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="frequent_object",
#     num_options=4,
#     seed=42,
#     generator_config={"frequency_type": "most"}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/frequent_object_most")
# print()

# # Test 16: frequent_object (least)
# print("="*70)
# print("Testing: frequent_object (least)")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="frequent_object",
#     num_options=4,
#     seed=42,
#     generator_config={"frequency_type": "least"}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/frequent_object_least")
# print()

# # Test 17: list_attribute_frequent (most)
# print("="*70)
# print("Testing: list_attribute_frequent (most)")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="list_attribute_frequent",
#     num_options=4,
#     seed=42,
#     generator_config={"frequency_type": "most"}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/list_attribute_frequent")
# print()

# # Test 18: count_attribute_frequent (least)
# print("="*70)
# print("Testing: count_attribute_frequent (least)")
# print("="*70)
# task_plan = TaskPlan(
#     generator_type="count_attribute_frequent",
#     num_options=4,
#     seed=42,
#     generator_config={"frequency_type": "least"}
# )
# generator.generate(task_plan, num_tasks=5, output_dir="./output_test/count_attribute_frequent")
# print()


print("="*70)
print("✅ All tests completed!")
print("="*70)
