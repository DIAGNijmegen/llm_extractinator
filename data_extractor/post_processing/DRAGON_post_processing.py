from pathlib import Path
from typing import List
import argparse
import json


def save_json(data: dict, filepath: Path) -> None:
    """
    Saves a dictionary to a JSON file.
    """
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)


def print_processing_message(task_id: str) -> None:
    """
    Prints a message indicating the task being processed.
    """
    print(f"Post-processing Task{task_id}...")


def process_file(filepath: Path, task_id: str) -> None:
    """
    Post-processes a single file to work for the DRAGON evaluation.
    """

    with open(filepath, "r") as file:
        data = json.load(file)

    backup_filename = filepath.with_name(f"{filepath.stem}_original_predictions.json")
    if not backup_filename.exists():
        save_json(
            data=data,
            filepath=backup_filename,
        )

    binary_class_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    binary_class_ids = [f"{int(class_id):03}" for class_id in binary_class_ids]

    multi_class_ids = [9, 10, 11, 12, 13, 14]
    multi_class_ids = [f"{int(class_id):03}" for class_id in multi_class_ids]

    single_regression_ids = [19, 20, 21, 22, 23]
    single_regression_ids = [
        f"{int(class_id):03}" for class_id in single_regression_ids
    ]

    if task_id in binary_class_ids:
        print_processing_message(task_id)
        try:
            for example in data:
                example["single_label_binary_classification"] = example.pop("label")
        except KeyError:
            print(f"Task {task_id} does not contain 'label' key.")
            pass
        save_json(data=data, filepath=filepath)
    elif task_id in multi_class_ids:
        print_processing_message(task_id)
        try:
            for example in data:
                example["single_label_multi_class_classification"] = example.pop(
                    "label"
                )
        except KeyError:
            print(f"Task {task_id} does not contain 'label' key.")
            pass
        save_json(data=data, filepath=filepath)
    elif task_id in single_regression_ids:
        print_processing_message(task_id)
        try:
            for example in data:
                example["single_label_regression"] = example.pop("label")
        except KeyError:
            print(f"Task {task_id} does not contain 'label' key.")
            pass
        save_json(data=data, filepath=filepath)
    elif task_id == "015":
        print_processing_message(task_id)
        try:
            for example in data:
                example["multi_label_binary_classification"] = [
                    example.pop("biopsy"),
                    example.pop("cancer"),
                    example.pop("high_grade_dysplasia"),
                    example.pop("hyperplastic_polyps"),
                    example.pop("low_grade_dysplasia"),
                    example.pop("non_informative"),
                    example.pop("serrated_polyps"),
                ]
        except KeyError:
            print(f"Task {task_id} does not contain the correct keys.")
            pass
        save_json(data=data, filepath=filepath)
    elif task_id == "016":
        print_processing_message(task_id)
        try:
            for example in data:
                example["multi_label_binary_classification"] = [
                    example.pop("lesion_1"),
                    example.pop("lesion_2"),
                    example.pop("lesion_3"),
                    example.pop("lesion_4"),
                    example.pop("lesion_5"),
                ]
        except KeyError:
            print(f"Task {task_id} does not contain the correct keys.")
            pass
        save_json(data=data, filepath=filepath)
    elif task_id == "017":
        print_processing_message(task_id)
        try:
            for example in data:
                example["multi_label_multi_class_classification"] = [
                    example.pop("attenuation"),
                    example.pop("location"),
                ]
        except KeyError:
            print(f"Task {task_id} does not contain the correct keys.")
            pass
        save_json(data=data, filepath=filepath)
    elif task_id == "018":
        print_processing_message(task_id)
        try:
            for example in data:
                example["multi_label_multi_class_classification"] = [
                    example.pop("left"),
                    example.pop("right"),
                ]
        except KeyError:
            print(f"Task {task_id} does not contain the correct keys.")
            pass
        save_json(data=data, filepath=filepath)
    elif task_id == "024":
        print_processing_message(task_id)
        try:
            for example in data:
                example["multi_label_regression"] = [
                    example.pop("lesion_1"),
                    example.pop("lesion_2"),
                    example.pop("lesion_3"),
                    example.pop("lesion_4"),
                    example.pop("lesion_5"),
                ]
        except KeyError:
            print(f"Task {task_id} does not contain the correct keys.")
            pass
        save_json(data=data, filepath=filepath)


def post_process(output_path: Path, task_ids: List[int]) -> None:
    """
    Post-processes the predictions generated by the prediction task to work for the DRAGON evaluation.
    """
    for task_id in task_ids:
        task_id = f"{int(task_id):03}"
        task_folders = list(output_path.glob(f"Task{task_id}_*"))
        assert len(task_folders) == 5

        for task_folder in task_folders:
            filepath = task_folder / "nlp-predictions-dataset.json"
            process_file(filepath=filepath, task_id=task_id)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process predictions for DRAGON evaluation."
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to the output directory of the prediction task.",
    )
    parser.add_argument(
        "--task_ids",
        type=int,
        nargs="+",
        help="Task IDs to post-process.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    post_process(output_path=args.output_path, task_ids=args.task_ids)


if __name__ == "__main__":
    main()
