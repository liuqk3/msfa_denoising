
import os

from os.path import join
# import csv
import json
from pyh import *


ratios = [10, 20, 40]
bands = [2] #list(range(25))

# img_type = 'noise_scale'
# img_dir = '../img_noisemodel'
# method_list = ['gt', 'ours_g', 'ours_Pg', 'ours_PgR', 'ours_PgRC', 'ours_PgRCU', 'ours_PgRCBU', 'urs_P', 'urs_R', 'urs_C', 'urs_U', 'urs_B']


img_type = 'band'
img_dir = '../img_noisemodel_raw'
method_list = ['gt', 'real_data', 'DANet', 'PgRCBU', 'complex', 'PgRCbU', 'g', 'pg', 'PgC'] 

for ratio in ratios:
    for band in bands: 
        image_list = list(range(50)) 

        apache_path = 'htmls' 
        os.makedirs(apache_path, exist_ok=True)
        html_path = 'html_different_noise_ratio_{}_band_{}_img_{}.html'.format(ratio, band, img_type) 


        page = PyH('different type of noise with ratio: {}, band: {}, img: {}'.format(ratio, band, img_type))
        # page << h1('Models tested on object mask.', cl='center')
        page << style(type="text/css")
        page << div(id='content')
        page.content << table(id='table', border=0, style="table-layout: fixed; text-align:center; padding-top: 0px;")
        page.content.table << tbody(id='tbody')

        cnt = 0

        for image in image_list:
            image = str(image)
            id = 'image'+str(cnt)
            page.content.table.tbody << tr(id=id)
            # completed image 
            for method_name in method_list:
                image_label = td(haligh="center", style="word-wrap: break-word;", valigh="top")
                if method_name == 'gt':
                    img_path = os.path.join(img_dir, method_name, str(ratio), image, '{}_{}.png'.format('band', band))
                else:
                    img_path = os.path.join(img_dir, method_name, str(ratio), image, '{}_{}.png'.format(img_type, band))
                # import pdb; pdb.set_trace()
                # assert os.path.exists(img_path), 'image not exists!'
                image_label << img(src=img_path, alt="None", style="width:250px; margin:2px;")
                page.content.table.tbody.__dict__[id] << image_label

            page.content.table.tbody.__dict__[id] <<td(image, colspan="1", halign="center", scope="colgroup", style="word-wrap: break-word;", valign="center")

            # image title
            page.content.table.tbody << tr(id=id+'title')
            for method_name in method_list:
                text = str(method_name)
                page.content.table.tbody.__dict__[id+'title'] << td(text, colspan="1", halign="center", scope="colgroup", style="word-wrap: break-word;", valign="center")

            cnt += 1

            # if limited_num != -1 and cnt > limited_num:
            #     break


        save_path = os.path.join(apache_path, html_path)
        with open(save_path, 'w') as f:
            f.write(page.render())
        print('saved to {}'.format(save_path))