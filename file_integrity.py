import os
import torch

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
            response = input("Label-Image Sequencing fails, force start? [y/n]: ")
            if response.lower() == 'y':
                print("\nForcing start despite mismatched files.\n")
            else:
                raise ValueError("Stopping due to Label-Image Sequence failure.")
        else:
            print("\nForcing start despite mismatched files.\n")

def isGPU(force_start=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
    else:
        if force_start == False:
            response = input("GPU is not mounted, continue with CPU only? [y/n]:")
            if response.lower() == 'y':
                print("CUDA is not available. Using CPU.")
            else:
                raise ValueError("Stopping due to GPU not being mounted")
        else:
            print("CUDA is not available. Using CPU.")