import matplotlib.patches as patches
import matplotlib.pyplot as plt


def draw_bounding_boxes(image, annotations):
    """
    Draws bounding boxes on the image based on annotation data.

    Parameters:
    - image: The image as a numpy array.
    - annotations: List of annotations, where each annotation contains 'bbox'.
    """
    # Create a figure and axis for the image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # For each annotation, add a rectangle for the bounding box
    for ann in annotations:
        # Bounding box [x_min, y_min, width, height]
        bbox = ann["bbox"]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.show()
