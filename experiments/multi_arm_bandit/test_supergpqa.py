"""
Test SuperGPQA integration.

This script verifies that the SuperGPQA loader and answer selector work correctly.
"""

import sys
import os

# Add both experiments and project root to path
experiments_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(experiments_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, experiments_dir)

from multi_arm_bandit import (
    SuperGPQADataset,
    extract_answer_letter,
    evaluate_supergpqa_answer,
    format_supergpqa_prompt,
)


def test_answer_extraction():
    """Test answer letter extraction from various response formats."""
    print("Testing answer letter extraction...")

    test_cases = [
        ("A) This is the correct answer", "A"),
        ("The answer is B", "B"),
        ("I believe option C is correct", "C"),
        ("Answer: D", "D"),
        ("Just E", "E"),
        ("F.", "F"),
        ("The correct choice is G because...", "G"),
        ("Random text with H in it", "H"),
    ]

    passed = 0
    for response, expected in test_cases:
        result = extract_answer_letter(response)
        if result == expected:
            print(f"  ✓ '{response[:30]}...' → {result}")
            passed += 1
        else:
            print(f"  ✗ '{response[:30]}...' → {result} (expected {expected})")

    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_dataset_loading():
    """Test loading mock SuperGPQA dataset."""
    print("\nTesting dataset loading...")

    dataset = SuperGPQADataset.load_mock_dataset(num_examples=20)

    print(f"  Loaded {len(dataset)} examples")

    if len(dataset) != 20:
        print(f"  ✗ Expected 20 examples, got {len(dataset)}")
        return False

    # Check first example
    example = dataset.examples[0]

    print(f"  First example:")
    print(f"    Subject: {example.subject}")
    print(f"    Question: {example.question[:60]}...")
    print(f"    Options: {len(example.options)}")
    print(f"    Answer: {example.answer}")

    if not example.question:
        print(f"  ✗ Example has no question")
        return False

    if len(example.options) < 4:
        print(f"  ✗ Example has too few options: {len(example.options)}")
        return False

    print(f"  ✓ Dataset loaded successfully")
    return True


def test_prompt_formatting():
    """Test formatting SuperGPQA questions as prompts."""
    print("\nTesting prompt formatting...")

    dataset = SuperGPQADataset.load_mock_dataset(num_examples=5)
    example = dataset.examples[0]

    prompt = format_supergpqa_prompt(example)

    print(f"  Formatted prompt length: {len(prompt)} characters")
    print(f"  Prompt preview:")
    print("  " + "-" * 50)
    for line in prompt.split("\n")[:10]:
        print(f"  {line}")
    print("  " + "-" * 50)

    # Check prompt contains key elements
    checks = [
        ("Question:" in prompt, "Contains 'Question:'"),
        ("Options:" in prompt, "Contains 'Options:'"),
        ("Answer:" in prompt, "Contains 'Answer:'"),
        (example.question in prompt, "Contains question text"),
        (len([opt for opt in example.options if opt in prompt]) >= len(example.options),
         "Contains all options"),
    ]

    all_passed = True
    for check, desc in checks:
        if check:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc}")
            all_passed = False

    return all_passed


def test_evaluation():
    """Test answer evaluation."""
    print("\nTesting answer evaluation...")

    test_cases = [
        ("A", "A", True),
        ("B", "B", True),
        ("The answer is C", "C", True),
        ("I think D is correct", "D", True),
        ("A", "B", False),
        ("The answer is B", "C", False),
    ]

    passed = 0
    for prediction, correct, expected_result in test_cases:
        result = evaluate_supergpqa_answer(prediction, correct)
        if result == expected_result:
            status = "✓" if result else "✗ (correct)"
            print(f"  {status} '{prediction}' vs '{correct}' → {result}")
            passed += 1
        else:
            print(f"  ✗ '{prediction}' vs '{correct}' → {result} (expected {expected_result})")

    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_sampling():
    """Test dataset sampling."""
    print("\nTesting dataset sampling...")

    dataset = SuperGPQADataset.load_mock_dataset(num_examples=20)

    # Sample with seed for reproducibility
    sample1 = dataset.sample(5, seed=42)
    sample2 = dataset.sample(5, seed=42)

    if len(sample1) != 5:
        print(f"  ✗ Expected 5 samples, got {len(sample1)}")
        return False

    # Check reproducibility
    same = all(
        s1.question == s2.question
        for s1, s2 in zip(sample1, sample2)
    )

    if same:
        print(f"  ✓ Sampling is reproducible with seed")
    else:
        print(f"  ✗ Sampling not reproducible")
        return False

    # Check different seeds give different samples
    sample3 = dataset.sample(5, seed=100)
    different = any(
        s1.question != s3.question
        for s1, s3 in zip(sample1, sample3)
    )

    if different:
        print(f"  ✓ Different seeds produce different samples")
    else:
        print(f"  ✗ Different seeds produce same samples")
        return False

    print(f"  ✓ Sampling works correctly")
    return True


def run_all_tests():
    """Run all SuperGPQA tests."""
    print("\n" + "=" * 60)
    print("SuperGPQA Integration Tests")
    print("=" * 60)

    tests = [
        ("Answer Extraction", test_answer_extraction),
        ("Dataset Loading", test_dataset_loading),
        ("Prompt Formatting", test_prompt_formatting),
        ("Answer Evaluation", test_evaluation),
        ("Dataset Sampling", test_sampling),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")

    if total_passed == len(results):
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED ✗")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
