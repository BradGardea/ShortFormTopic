import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import logging
import cv2
import numpy as np
import torch
from torch import nn

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.sf = sf
        print([in_nc, out_nc, nf, nb, gc, sf])

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.sf==4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf==4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

def imread_uint(path, n_channels=3):
    """Read an image and return it as a NumPy array (uint8)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if n_channels == 3 and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif n_channels == 1 and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def uint2tensor4(img):
    """Convert a uint8 NumPy array to a 4D tensor."""
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = torch.from_numpy(img).float() / 255.0  # Scale to [0, 1]
    return img.unsqueeze(0)  # Add batch dimension

def tensor2uint(tensor):
    """Convert a 4D tensor to a uint8 NumPy array."""
    img = tensor.squeeze(0).clamp(0, 1).cpu().numpy()  # Remove batch, clamp to [0, 1]
    img = np.transpose(img, (1, 2, 0))  # CHW to HWC
    return (img * 255.0).round().astype(np.uint8)

def imsave(img, path):
    """Save an image (NumPy array) to a file."""
    cv2.imwrite(path, img)

def process_video(input_video_path, output_video_path, model, device, fps=30):
    """Process each frame of a video using the model and save the result as a video."""
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {input_video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    video_fps = fps if fps else original_fps

    # Read the first frame to determine frame dimensions after processing
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read the first frame of the video.")
    
    # Process the first frame to determine output dimensions
    processed_frame = process_video_frame(frame, model, device)
    processed_height, processed_width = processed_frame.shape[:2]

    # Ensure output directory exists
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize VideoWriter with processed frame dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_video_path, fourcc, 4, (processed_width, processed_height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to create output video file: {output_video_path}")
    
    # Write the first processed frame to the output video
    out.write(processed_frame)

    # Process the remaining frames
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"Processing frame {frame_idx + 1}")
        processed_frame = process_video_frame(frame, model, device)
        out.write(processed_frame)
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved to: {output_video_path}")


def process_video_frame(frame, model, device):
    """Apply the model to a single video frame."""
    # Convert frame (BGR image) to RGB and normalize
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32)

    # Apply the model
    with torch.no_grad():
        processed_tensor = model(frame_tensor)

    # Convert the processed tensor back to a NumPy array
    processed_frame = (
        processed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255
    ).astype("uint8")
    
    # Convert RGB back to BGR for OpenCV compatibility
    return cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    
    # User-defined variables
    file_name = "generated_video_00009"
    model_path = "model_zoo/BSRGAN.pth"  # Path to the model
    input_video_path = f"playground\\{file_name}.mp4"  # Path to the input video
    output_video_path = f"playground\\{file_name}_output_video.mp4"  # Path to save the output video
    
    # Ensure paths exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Define the model
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)  # Define network
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    torch.cuda.empty_cache()
    
    # Process the video
    process_video(input_video_path, output_video_path, model, device)

if __name__ == '__main__':
    main()