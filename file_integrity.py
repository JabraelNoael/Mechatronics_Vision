import os

def checkDataSequencing(force_start=False):
    force_start = force_start
    images_dir = "data/images/train"
    labels_dir = "data/labels/train"

    # Get all file stems (filenames without extensions)
    images = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))}
    labels = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))}

    # Find mismatches
    extra_images = images - labels
    extra_labels = labels - images

    # Report results
    if not extra_labels and not extra_images:
        print("Label-Image Sequencing is Good")
    else:
        print("Mismatched files found:")
        if extra_images:
            print(f"\t- Extra image file(s): {sorted(extra_images)}")
        if extra_labels:
            print(f"\t- Extra label file(s): {sorted(extra_labels)}")
        if force_start == False:
            raise ValueError("Cannot proceed with mismatched label-image types")
        else:
            print("\nForcing start despite mismatched files.\n")