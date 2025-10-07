import torch

#####################################################################################################
################################### Create Segmentaion Dataset ######################################
##################################################################################################### 

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, image_shape,
                 min_num_per_image = 1, max_num_per_image = 1, 
                 max_iou = 0.05, n_samples = None, transforms=None, rotate=False) -> None:

        
        self.dim = len(image_shape)-1
        self.dataset = dataset
        self.image_shape = image_shape
        self.min_num_per_image = min_num_per_image
        self.max_num_per_image = max_num_per_image
        self.max_iou = max_iou
        self.transforms = transforms
        self.rotate = rotate
        self.n_samples = len(self.dataset) if n_samples is None else n_samples
        
        if len(self.image_shape) != len(dataset[0][0].shape):
            raise ValueError("`image_shape` and images in the dataset should have same number of dimensions.")

    def __getitem__(self, index):
        if 0 <= index < self.n_samples:
            image, target = self.create_semantic_segmentation_data()
        else:
            raise ValueError("Index Out of Bounds.")

        if self.transforms is not None:
            image, target = self.transforms(image, target)
   
        return image, target

    def __len__(self):
        return self.n_samples

    
    def create_semantic_segmentation_data(self):

        num_digits = torch.randint(self.min_num_per_image, self.max_num_per_image + 1, (1,)).item()

        input_array, arrays_overlaid, labels_overlaid, bounding_boxes_overlaid = self.overlay_arrays(
            num_input_arrays_to_overlay=num_digits)



        target_array = self.create_segmentation_target(images=arrays_overlaid,
                                                    labels=labels_overlaid,
                                                    bounding_boxes=bounding_boxes_overlaid)


        return input_array, target_array


    def create_segmentation_target(self, images, labels, bounding_boxes):
        if len(bounding_boxes) != len(labels) != len(images):
            raise ValueError(
                f'The length of bounding_boxes must be the same as the length of labels. Received shapes: {bounding_boxes.shape}!={labels.shape}')

        # Adding one for background
        individual_targets = [torch.zeros(self.image_shape) for _ in range(len(images)+1)]
        labels.insert(0,-1)
        labels = torch.tensor(labels, dtype=torch.int32) + 1
        for i in range(len(bounding_boxes)):
            slices = tuple(slice(dmin, dmax) for dmin, dmax in zip(bounding_boxes[i][0].tolist(), bounding_boxes[i][1].tolist()))
            individual_targets[i+1][(Ellipsis,) + slices] += images[i]

        targets = torch.stack(individual_targets, dim=0)
        target = labels[torch.argmax(torch.mean(targets, dim=1), dim=0)]

        return target
    
    def overlay_arrays(self, num_input_arrays_to_overlay: int):

        output_array = torch.zeros(self.image_shape)

        indices = torch.randint(0, len(self.dataset), (num_input_arrays_to_overlay, ))
        bounding_boxes = []
        bounding_boxes_as_tuple = []
        arrays_overlaid = []
        labels_overlaid = []
        
        for i in indices:
            images, labels = self.dataset[i]
            bounding_box = self.overlay_at_random(array1=output_array, array2=images, bounding_boxes=bounding_boxes)
            
            if bounding_box is None:
                break
            
            arrays_overlaid.append(images)
            bounding_boxes_as_tuple.append(list(bounding_box.values()))
            bounding_boxes.append(bounding_box)
            try:
                labels_overlaid.append(labels.item())
            except AttributeError:
                labels_overlaid.append(labels)
            
            
        arrays_overlaid = arrays_overlaid
        labels_overlaid = labels_overlaid
        bounding_boxes_overlaid = bounding_boxes_as_tuple

        return output_array, arrays_overlaid, labels_overlaid, bounding_boxes_overlaid

    def overlay_at_random(self, array1, array2, bounding_boxes = None):
        if not bounding_boxes:
            bounding_boxes = []

        dimension1 = torch.tensor(array1.shape[-self.dim:])
        dimension2 = torch.tensor(array2.shape[-self.dim:])
        
        is_valid = False
        # This number is arbitrary. There are better ways of doing this but this is fast enough.
        max_attempts = 1000
        attempt = 0
        while not is_valid:
            if attempt > max_attempts:
                return
            else:
                attempt += 1
            candidate = torch.tensor([torch.randint(0, max_dim.item()+1, (1,)).item() 
                                      for max_dim in dimension1-dimension2])
            
            candidate_bounding_box = {
                'min' : candidate,
                'max' : candidate + dimension2
            }
    
            is_valid = True
            for bounding_box in bounding_boxes:
                iou = self.calculate_iou(bounding_box, candidate_bounding_box)
                if  iou > self.max_iou:
                    is_valid = False
                    break

        self.overlay_array(array1=array1, array2=array2, candidate=candidate)

        return candidate_bounding_box
    
    def overlay_array(self, array1, array2, candidate) -> torch.Tensor:

        other1, dimension1 = torch.tensor(array1.shape[:-self.dim]), torch.tensor(array1.shape[-self.dim:])
        other2, dimension2 = torch.tensor(array2.shape[:-self.dim]), torch.tensor(array2.shape[-self.dim:])

        if torch.any(dimension1<dimension2):
            raise ValueError('array2 must have a smaller shape than array1')

        if torch.any(other1 != other2):
            raise ValueError('array1 and array2 must have same non-singleton dimension.')

        max_array_value = max([torch.max(array1), torch.max(array2)])
        min_array_value = min([torch.min(array1), torch.min(array2)])
        
        slices = tuple(slice(dmin, dmax) for dmin, dmax in zip(candidate.tolist(), (candidate+dimension2).tolist()))
        array1[(Ellipsis,) + slices] += array2
        array1 = torch.clamp_(array1, min_array_value, max_array_value)

        return
    
    
    def calculate_iou(self, bounding_box1: dict, bounding_box2: dict) -> float:
        A1 = torch.prod(bounding_box1['max'] - bounding_box1['min'])
        A2 = torch.prod(bounding_box2['max'] - bounding_box2['min'])
        
        intersection_min = torch.max(torch.stack([bounding_box1['min'], bounding_box2['min']], dim=0), dim=0)[0]
        intersection_max = torch.min(torch.stack([bounding_box1['max'], bounding_box2['max']], dim=0), dim=0)[0]
        
        iou = 0 if torch.any(intersection_max<intersection_min) else torch.prod(intersection_max-intersection_min) / (A1 + A2)
        return iou