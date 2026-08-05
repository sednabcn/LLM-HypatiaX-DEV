import os

with open("json_files.txt", "w") as f:
    f.write(
        "\n".join(
            os.path.join(root, file)
            for root, dirs, files in os.walk(".")
            for file in files
            if file.startswith("all_") and file.endswith(".json")
        )
    )
