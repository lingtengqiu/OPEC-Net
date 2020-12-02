from ..registry import PIPELINES
import torch
import numpy as np
import cv2
from engineer.utils.tensor_np import to_torch,torch_to_im,im_to_torch
from engineer.utils.metric import get_3rd_point
@PIPELINES.register_module
class crop_large(object):
    def __init__(self,RGB,inputResH,inputResW):
        super(crop_large,self).__init__()
        self.RGB = RGB
        self.resH,self.resW = inputResH, inputResW
    def __call__(self,results):
        '''

        :param results:
        orig_img, im_name, boxes, scores, inps, pt1, pt2
        :return:
        (inps, orig_img, im_name, boxes, scores, pt1, pt2)
        '''
        orig_img = results['orig_img']
        im_name = results['im_name']
        boxes = results['boxes']
        scores = results['scores']
        inps = results['inps']
        pt1 = results['pt1']
        pt2 = results['pt2']
        if boxes is None or boxes.nelement() == 0:
            results['inps'] = None
            results['pt1'] =None
            results['pt2'] = None
        else:

            inp = self.im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            inps, pt1, pt2 = self._crop_from_dets_train_single(inp, boxes, inps, pt1, pt2)
            results['inps'] = inps
            results['pt1'] = pt1
            results['pt2'] = pt2
        return results

    def im_to_torch(self,img):
        img = np.transpose(img, (2, 0, 1))  # C*H*W
        img = to_torch(img).float()
        if img.max() > 1:
            img /= 255
        return img

    def _crop_from_dets_train_single(self,img, boxes, inps, pt1, pt2):
        '''
        Crop human from origin image according to Dectecion Results
        '''

        tmp_img = img

        # to subtract mean RGB 0.406 0.457 0.480
        R,G,B = self.RGB
        tmp_img[0].add_(R)
        tmp_img[1].add_(G)
        tmp_img[2].add_(B)
        for i, box in enumerate(boxes):
            upLeft = torch.Tensor(
                (float(box[0]), float(box[1])))
            bottomRight = torch.Tensor(
                (float(box[2]), float(box[3])))
            try:
                inps[i] = self.cropBox(tmp_img.clone(), upLeft, bottomRight)
            except IndexError:
                print(tmp_img.shape)
                print(upLeft)
                print(bottomRight)
                print('===')
            pt1[i] = upLeft
            pt2[i] = bottomRight
        return inps, pt1, pt2

    def cropBox(self,img, ul, br):
        ul = ul.int()
        br = (br - 1).int()
        # br = br.int()
        lenH = max((br[1] - ul[1]).item(), (br[0] - ul[0]).item() * self.resH / self.resW)
        lenW = lenH * self.resW / self.resH
        if img.dim() == 2:
            img = img[np.newaxis, :]

        box_shape = [(br[1] - ul[1]).item(), (br[0] - ul[0]).item()]
        pad_size = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) // 2]
        # Padding Zeros
        if ul[1] > 0:
            img[:, :ul[1], :] = 0
        if ul[0] > 0:
            img[:, :, :ul[0]] = 0
        if br[1] < img.shape[1] - 1:
            img[:, br[1] + 1:, :] = 0
        if br[0] < img.shape[2] - 1:
            img[:, :, br[0] + 1:] = 0

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)

        src[0, :] = np.array(
            [ul[0] - pad_size[1], ul[1] - pad_size[0]], np.float32)
        src[1, :] = np.array(
            [br[0] + pad_size[1], br[1] + pad_size[0]], np.float32)
        dst[0, :] = 0
        dst[1, :] = np.array([self.resW - 1, self.resH - 1], np.float32)

        src[2:, :] = get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        dst_img = cv2.warpAffine(torch_to_im(img), trans,
                                 (self.resW, self.resH), flags=cv2.INTER_LINEAR)

        return im_to_torch(torch.Tensor(dst_img))
    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str