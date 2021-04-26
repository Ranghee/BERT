#################################################

# BERT made by kangheelee                       #

#################################################


### 파라미터 설명

'''

corpus - 입력 파일
output - 저장 파일
n_seq - max_token_length
vocab - vocab file
mask_prob - mask 확률 논문 : 0.15

'''

import argparse, os, json
import sentencepiece as spm
from tqdm import tqdm
from random import randrange, random, choice, shuffle



def make_mask(tokens, mask_cnt, vocab_list):
    token_idx = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        if 0 < len(token_idx) and not token.startswtih(u"\u2581"):
            token_idx[-1].append(i)
        else:
            token_idx.append([i])
    shuffle(token_idx)

    mask_lms = []
    for index_set in token_idx:
        if len(mask_lms) >= mask_cnt:
            break
        if len(mask_lms) + len(index_set) > mask_cnt:
            continue
        for index in index_set:
            masked_token = None
            if random() < 0.8:
                masked_token = "[MASK]"
            else:
                if random() < 0.5:
                    masked_token = tokens[index]
                else:
                    masked_token = choice(vocab_list)
            mask_lms.append({"index": index, "label": tokens[index]})  # 어디가 mask 되었는지 index 위치 정보
            tokens[index] = masked_token                               # 실제 mask 된 문장 데이터
    mask_lms = sorted(mask_lms, key=lambda  x : x["index"])
    mask_idx = [p["index"] for p in mask_lms]
    mask_label = [p["label"] for p in mask_lms]

    return tokens, mask_idx, mask_label


def cut_tokens(tokens_A, tokens_B, max_sequence):
    while True:
        total_len = len(tokens_A) + len(tokens_B)
        if total_len <= max_sequence:
            break
        if len(tokens_A) > len(tokens_B):
            del tokens_A[0]
        else:
            tokens_B.pop()


def process_ptr_data(data_list, idx, document, n_seq, mask_prob, vocab_list):
    max_sequence = n_seq - 3 # CLS, SEP, SEP
    target_sequence = max_sequence
    process_datas = []
    current_data = []
    current_len = 0
    for i in range(len(document)):   
        current_data.append(document[i]) 
        current_len += len(document[i])
        if current_len >= target_sequence or i == len(document) - 1:  # 현재 문장 길이가 512 - 3보다 작으면 512 - 3이 넘을때까지 그 다음 문장을 append 해줌 
            if 0 < len(current_data):
                A_end = 1
                if 1 < len(current_data):
                    A_end = randrange(1, len(current_data))        
                tokens_A = []
                for j in range(A_end):
                    tokens_A.extend(current_data[j])
                tokens_B = []
                if random() < 0.5 or 1 == len(current_data):    # 50퍼 확률로 is_not_next_sentence를 만듬
                    is_next = 0
                    ##### torkens_B_len = target_sequence - len(tokens_A)
                    random_doc_idx = idx
                    while idx == random_doc_idx:
                        random_doc_idx = randrange(0, len(data_list))
                    random_doc = data_list[random_doc_idx]      # doc 전체에서 현재 문장의 다음 문장이 아닌 랜덤으로 한 문장을 가져온다.
                    random_start = randrange(0, len(random_doc))
                    for j in range(random_start, len(random_doc)):
                        tokens_B.extend(random_doc[j])         # Token A 는 현재문장 Token B 는 NSP가 False인 문장 (50퍼의 확률로 정함)
                else:                                          # 50퍼 확률로 is_next_sentence를 만듬
                    is_next = 1
                    for j in range(A_end, len(current_data)):
                        tokens_B.extend(current_data[j])
                cut_tokens(tokens_A, tokens_B, max_sequence) # max_sequence_len 보다 token_A와 token_B의 합이 더 길면  Token_A가 Token_B보다 더 길면 Token_A의 앞을 자르고 반대면 Token_B의 뒤를 잘르며 max_sequence_len과 같을때 까지 반복한다.
                assert 0 < len(tokens_A)
                assert 0 < len(tokens_B)

                tokens = ["[CLS]"] + tokens_A + ["[SEP]"] + tokens_B + ["[SEP]"]
                seg = [0] * (len(tokens_A) + 2) + [1] * (len(tokens_B) + 1)  # CLS + token_A + SEP 는  00000000 으로     token_B + SEP 는 11111111로 치환해서 00000000111111111 로 segment data를 만들어준다.
                
                tokens, mask_idx, mask_label = make_mask(tokens, int((len(tokens) - 3) * mask_prob), vocab_list)

                process_data = {
                    "tokens": tokens,
                    "segment": seg,
                    "is_next": is_next,
                    "mask_idx": mask_idx,
                    "mask_label": mask_label
                }
                process_datas.append(process_data)
            
            current_data = []
            current_len = 0
    return process_datas


def load_data(corpus_data, vocab):
    cnt = 0

    with open(corpus_data, "r") as c_f:
        for line in c_f:
            cnt += 1
    document_list = []

    with open(corpus_data, "r") as f:
        document = []
        for _ , line in enumerate(tqdm(f, total=cnt, desc="Loading corpus data", unit=" 라인")):
            line = line.strip()
            if line == "":
                if 0 < len(document):
                    document_list.append(document)
                    document = []
            else:
                pieces = vocab.encode_as_pieces(line)
                if 0 < len(pieces):
                    document.append(pieces)
        if document:
            document_list.append(document)

    return document_list


def make_ptr_data(args):
    vocab = spm.SentencePieceProcessor()
    vocab.load(args.vocab)
    vocab_list = []
    for id in range(vocab.get_piece_size()):
        if not vocab.is_unknown(id):
            vocab_list.append(vocab.id_to_piece(id))

    data_list = load_data(args.corpus, vocab)
    for dupe in range(args.dupe_factor):
        output = args.output.format(dupe)
        if os.path.isfile(output):
            continue
        with open(output, "w") as out:
            for i, document in enumerate(tqdm(data_list, desc="Create output data NOW", unit=" 라인")):
                process_datas = process_ptr_data(data_list, i, document, args.n_seq, args.mask_prob, vocab_list)
                for process_data in process_datas:
                    out.write(json.dumps(process_data))
                    out.write("\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus", default="data/corpus.txt", type=str, required=False, help="input corpus file")
    parser.add_argument("--output", default="data/ptr_input_data.json", type=str, required=False, help="output file - corpus process to pretraining input data")
    parser.add_argument("--n_seq", default=512, type=int, required=False, help="number of max token")
    parser.add_argument("--vocab", default="data/vocab.model", type=str, required=False, help="spm vocab file")
    parser.add_argument("--mask_prob", default="0.15", type=float, required=False, help="mask ratio - in the paper : 0.15")
    parser.add_argument("--dupe_factor", default=10, type=int, required=False, help="Number of different mask corpus data")
    args = parser.parse_args()

    if not os.path.isfile(args.output):
        make_ptr_data(args)
    else:
        print("Same of output_data_file's name is already exist")


