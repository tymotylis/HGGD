
import torch.nn.functional as F
import numpy as np
import torch
import cv2


class BoundingBox():
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
    
    def get_width(self):
        return self.x_max - self.x_min

    def get_height(self):
        return self.y_max - self.y_min

    def clip_img(self, img):
        return img[self.y_min:self.y_max, self.x_min:self.x_max]

    def clip_model_input(self, img):
        return img[:, :, self.x_min:self.x_max, self.y_min:self.y_max]
    
    def get_rescaled(self, old_image_dimensions, new_image_dimensions):
        scalar = new_image_dimensions.x_max / old_image_dimensions.x_max
        return BoundingBox(round(self.x_min * scalar), round(self.y_min * scalar), round(self.x_max * scalar), round(self.y_max * scalar))
    
    def clip_to(self, bb):
        self.x_min = self.clamp(self.x_min, bb.x_min, bb.x_max)
        self.x_max = self.clamp(self.x_max, bb.x_min, bb.x_max)
        self.y_min = self.clamp(self.y_min, bb.y_min, bb.y_max)
        self.y_max = self.clamp(self.y_max, bb.y_min, bb.y_max)

        return self

    def copy(self):
        return BoundingBox(self.x_min, self.y_min, self.x_max, self.y_max)

    def with_padding(self, padding, image_bb):
        the_copy = self.copy()

        the_copy.x_min -= padding[0]
        the_copy.y_min -= padding[1]
        the_copy.x_max += padding[0]
        the_copy.y_max += padding[1]

        the_copy.clip_to(image_bb)

        return the_copy

    def clamp(self, number, lower, upper):
        return int(min(max(number, lower), upper))
    
    def __repr__(self):
        return "%s:%s to %s:%d" % (self.x_min, self.y_min, self.x_max, self.y_max)
    
    def __str__(self):
        return self.__repr__()


class Cell():

    def __init__(self, tile_coords, tile_count, tile_dimensions, padding, image, foreground_mask, scale_factor = 1):
        if scale_factor != 1:
            tile_dimensions = [int(tile_dimensions[0] * scale_factor), int(tile_dimensions[1] * scale_factor)]
            padding = [int(padding[0] * scale_factor), int(padding[1] * scale_factor)]

            # Downscale image tensor
            image = F.interpolate(
                image,
                scale_factor=scale_factor,
                mode='bilinear',
                align_corners=False
            )
            print(image.shape)

            # Downscale foreground mask
            foreground_mask = cv2.resize(
                foreground_mask,
                dsize=None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_NEAREST
            )

        self.tile_dimensions = tile_dimensions
        self.original_image_bb = BoundingBox(0, 0, image.shape[2], image.shape[3])
        self.original_bb = self.get_bb(tile_coords, tile_dimensions)

        self.adjusted_bb = self.clip_background(foreground_mask, self.original_bb)

        self.model_input = None

        if self.adjusted_bb != None:
            self.adjusted_bb_padded = self.adjusted_bb.with_padding(padding, self.original_image_bb)
            self.model_input = self.adjusted_bb_padded.clip_model_input(image)

    def fit_output_to_size(self, model_output, target_shape, location_map):
        target_shape_bb = BoundingBox(0, 0, target_shape[2], target_shape[3])

        original_bb_rescaled = self.original_bb.get_rescaled(self.original_image_bb, target_shape_bb)

        if model_output == None:
            return torch.full((target_shape[0], target_shape[1], original_bb_rescaled.get_width(), original_bb_rescaled.get_height()), -10)

        adjusted_bb_rescaled = self.adjusted_bb_padded.get_rescaled(self.original_image_bb, target_shape_bb)

        if not location_map:
            adjusted_clipped_bb_rescaled = BoundingBox(
                adjusted_bb_rescaled.x_min, 
                adjusted_bb_rescaled.y_min, 
                adjusted_bb_rescaled.x_max, 
                adjusted_bb_rescaled.y_max).clip_to(original_bb_rescaled)
        else:
            adjusted_clipped_bb_rescaled = self.adjusted_bb.get_rescaled(self.original_image_bb, target_shape_bb)

        if adjusted_clipped_bb_rescaled.get_width() == 0 or adjusted_clipped_bb_rescaled.get_height() == 0:
            return torch.full((target_shape[0], target_shape[1], original_bb_rescaled.get_width(), original_bb_rescaled.get_height()), -10)

        # Clip
        model_output = model_output[:, :,
            (adjusted_clipped_bb_rescaled.x_min - adjusted_bb_rescaled.x_min):
            (adjusted_clipped_bb_rescaled.x_max - adjusted_bb_rescaled.x_min),
            (adjusted_clipped_bb_rescaled.y_min - adjusted_bb_rescaled.y_min):
            (adjusted_clipped_bb_rescaled.y_max - adjusted_bb_rescaled.y_min),
            ]

        # Resize
        model_output = F.interpolate(model_output, size = (adjusted_clipped_bb_rescaled.get_width(), adjusted_clipped_bb_rescaled.get_height()))

        model_output = F.pad(
            model_output,
            (
                adjusted_clipped_bb_rescaled.y_min - original_bb_rescaled.y_min, 
                original_bb_rescaled.y_max - adjusted_clipped_bb_rescaled.y_max,
                adjusted_clipped_bb_rescaled.x_min - original_bb_rescaled.x_min, 
                original_bb_rescaled.x_max - adjusted_clipped_bb_rescaled.x_max
            ),
            value=-10
        )

        return model_output

    def get_bb(self, tile_coords, tile_dimensions):
        return BoundingBox(
            int(tile_coords[0] * tile_dimensions[0]), 
            int(tile_coords[1] * tile_dimensions[1]),
            int((tile_coords[0] + 1) * (tile_dimensions[0])),
            int((tile_coords[1] + 1) * tile_dimensions[1])).clip_to(self.original_image_bb)

    def clip_background(self, foreground_mask, designation):
        mask_cell = foreground_mask[designation.y_min:designation.y_max, designation.x_min:designation.x_max]

        ys, xs = np.where(mask_cell == 1)

        if len(xs) == 0:
            return None
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

        return BoundingBox(x_min + designation.x_min, y_min + designation.y_min, x_max + designation.x_min + 1, y_max + designation.y_min + 1)