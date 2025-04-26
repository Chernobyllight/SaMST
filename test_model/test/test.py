
import os
project_root = os.path.abspath('../..')
import sys
sys.path.append(project_root)

import yaml

from test_model import utils
from networks.transfer_net import TransformerNet

import torch
from torchvision import transforms

def stylize(opt):
    device = torch.device("cuda" if opt['cuda'] else "cpu")

    content_images = os.listdir(opt['content_image_dir'])
    style_model = TransformerNet(style_num=opt['style_num'])
    state_dict = torch.load(opt['model'])
    style_model.load_state_dict(state_dict)
    style_model.to(device)

    if not os.path.exists(opt['output_image_dir']):
        os.makedirs(opt['output_image_dir'])

    with torch.no_grad():
        for filename in content_images:

            print(filename)

            file_path = os.path.join(opt['content_image_dir'],filename)
            content_image = utils.load_image(filename=file_path, scale=opt['content_scale'])
            content_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])
            content_image = content_transform(content_image)
            content_image = content_image.unsqueeze(0).to(device)

            print(content_image.shape)

            for i in range(0, opt['style_num'] + 1):
                output, embedding = style_model(content_image, style_id=[i])
                output = output.cpu()
                utils.save_image(opt['output_image_dir'] + 'style' + str(i) + '_' + filename, output[0])






def main():

    with open('test.yml', 'r') as stream:
        opt = yaml.load(stream, Loader=yaml.FullLoader)

    stylize(opt)


if __name__ == "__main__":
    main()


