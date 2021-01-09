#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:55:11 2019

@author: verdunkelt
"""

#def beam_search(decoder, num_beams, max_len, *input):
#    """
#    a beam search implementation about seq2seq with attention
#    :param decoder:
#    :param num_beams: number of beam, int
#    :param max_len: max length of result
#    :param input: input of decoder
#    :return: list of index
#    """
#    # init
#    state = input[0]  # state of decoder
#    outputs = input[1]  # outputs of encoder
#    src_len = input[2]  # length of encode sequence
#    beams = [[[1], 1, state]]
#
#    cur_pro = 0
#    cur_seq = None
#    for i in range(max_len):
#        results = []
#        for beam in beams:
#            tgt = torch.LongTensor(beam[0][-1:]).unsqueeze(0).cuda()
#            input = [tgt, beam[2], outputs, src_len, 1]
#            output, state = decoder(*input)
#            v, i = torch.topk(output.view(-1).data, k=num_beams)
#            for m, n in zip(v, i):
#                gen_seq = beam[0] + [n.item()]
#                pro = beam[1] * m.item()
#                results.append([gen_seq, pro, state])
#
#                if n.item() == 2 and pro > cur_pro:  # eos_token = 2
#                    cur_pro = pro
#                    cur_seq = gen_seq
#
#        # filter beams
#        beams = []
#        for gen_seq, pro, state in results:
#            if pro > cur_pro:
#                beams.append([gen_seq, pro, state])
#        # cut
#        if len(beams) > num_beams:
#            results = []
#            pros = []
#            for beam in beams:
#                pros.append(beam[1])
#            pros_idx = np.array(pros).argsort()[-1*num_beams:]
#            for pro_idx in pros_idx:
#                results.append(beams[pro_idx])
#            beams = results
#
#        if len(beams) == 0:
#            return cur_seq
#
#    if cur_seq is not None:
#        return cur_seq
#    else:
#        max_pro = 0
#        max_seq = None
#        for beam in beams:
#            if beam[1] > max_pro:
#                max_pro = beam[1]
#                max_seq = beam[0]
#    return max_seq
