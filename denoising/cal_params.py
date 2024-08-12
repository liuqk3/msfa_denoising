import numpy as np

def cal_down(ch_in, ch_out):
    return ch_in*ch_out*4*4 #+ ch_out

def cal_conv_block(ch_in, ch_out):
    return ch_in*ch_out*3*3 + ch_out*ch_out*3*3 + ch_in*ch_out #+ 3*ch_out

def cal_up(ch_in, ch_out):
    return ch_in*ch_out*2*2 #+ ch_out

def cal(n1=64):
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

    total = 0
    total += cal_down(filters[0], filters[0])
    total += cal_down(filters[1], filters[1])
    total += cal_down(filters[2], filters[2])
    total += cal_down(filters[3], filters[3])

    total += cal_up(filters[4], filters[3])
    total += cal_up(filters[3], filters[2])
    total += cal_up(filters[2], filters[1])
    total += cal_up(filters[1], filters[0])

    total += cal_conv_block(3, filters[0])
    total += cal_conv_block(filters[0], filters[1])
    total += cal_conv_block(filters[1], filters[2])
    total += cal_conv_block(filters[2], filters[3])
    total += cal_conv_block(filters[3], filters[4])

    total += cal_conv_block(filters[4], filters[3])
    total += cal_conv_block(filters[3], filters[2])
    total += cal_conv_block(filters[2], filters[1])
    total += cal_conv_block(filters[1], filters[0])

    total += filters[0]*3*3*3 + 1
    print(total)

if __name__ == '__main__':
    cal(32)

