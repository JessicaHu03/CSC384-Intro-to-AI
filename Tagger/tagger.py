# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys
import numpy as np

# given possible tags and ambiguity tags
tags = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD", "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI", "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0", "UNC", "VBB", "VBD", "VBG", "VBI", "VBN", "VBZ", "VDB", "VDD",
        "VDG", "VDI", "VDN", "VDZ", "VHB", "VHD", "VHG", "VHI", "VHN", "VHZ", "VM0", "VVB", "VVD", "VVG", "VVI", "VVN", "VVZ", "XX0", "ZZ0", "AJ0-AV0", "AJ0-VVN", "AJ0-VVD", "AJ0-NN1", "AJ0-VVG", "AVP-PRP",  "AVQ-CJS", "CJS-PRP", "CJT-DT0", "CRD-PNI", "NN1-NP0", "NN1-VVB", "NN1-VVG", "NN2-VVZ", "VVD-VVN"]
training_size = 0
punc = []


def normalize(L):
    """ Normalize the probabilities in L """
    total = sum(L)
    for i in range(len(L)):
        p = L[i]
        percent = p / total
        L[i] = percent


def handle_unseen_words(tag_index, num_words, word, word_index, initial_prob, end_prob, freq_prob):
    """ Assign an emission prob to a word never seen before """
    prob = 0
    if word_index == 0:  # beginning of the sentence
        prob = initial_prob[tag_index] * \
            freq_prob[tag_index] * (1 / (training_size ** 5))
    elif word_index == num_words - 1:  # end of a sentence
        prob = end_prob[tag_index] * \
            freq_prob[tag_index] * (1 / (training_size ** 5))
    else:  # in the middle
        prob = freq_prob[tag_index] * (1 / (training_size ** 5))
    return prob


def viterbi(O, S, I, E, T, M, F):
    """
    Return the most likely path in a HMM given a sentence

    Inputs:
    O: observations array (words)
    S: hidden states (tags)
    I: initial probability      I[tag_index]
    E: end prob                 E[tag_index]
    T: transition prob          T[prev_tag_index][tag_index]
    M: Emission prob            M[tag_index][word]
    F: frequency prob           F[tag_index]

    Outputs:
    path: the most likely path of given observations
    prob: the probability of most likely states
    prev: the most likely previous state

    TODO: for a word that have encountered before but not in the desire tag
    TODO: check the path forward and backward with reverse transition prob
    TODO: for PUN and not PUN words
    TODO: prefix and suffix
    """
    # dimensions
    L = len(O)
    W = len(S)      # 76

    # prob and prev matrix
    prob = np.zeros([L, W])
    prev = np.zeros([L, W])

    # basecase (first word)
    for i in range(len(S)):
        # take care of the 0s in emit
        o = O[0].lower()
        if o not in M[i]:
            M[i][o] = handle_unseen_words(i, len(O), o, 0, I, E, F)
            # M[i][O[0]] = F[i] * 0.0000000001
        prob[0, i] = I[i] * M[i][o]
        prev[0, i] = np.NaN
    normalize(prob[0, :])
    # print(tags[np.argmax(prob[0, :])])

    # recursive step
    for t in range(1, len(O)):
        for i in range(len(S)):
            # take care of the 0s in emit
            o = O[t].lower()
            if o not in M[i]:
                M[i][o] = handle_unseen_words(i, len(O), o, t, I, E, F)
                # M[i][O[t]] = F[i] * 0.0000000001
            max_index = np.argmax(prob[t-1, :] * T[:][i] * M[i][o])
            # print(tags[max_index])
            prob[t, i] = prob[t-1, max_index] * T[max_index][i] * M[i][o]
            prev[t, i] = max_index
        normalize(prob[t, :])

    # the output path for this sentence
    path = ['' for _ in range(len(O))]
    last_max_index = np.argmax(prob[L - 1, :])
    path[-1] = tags[int(last_max_index)]
    # print("path[-1]: ", path[-1])
    for i in reversed(range(1, L)):
        new_max_index = prev[i, last_max_index]
        # print(tags[int(new_max_index)])
        path[i-1] = tags[int(new_max_index)]
        last_max_index = int(new_max_index)

    # print(path)
    return path


def HMM(train_data):  # checked
    """
    Construct the init_prob, end_prob, transit_prob, emit_prob, freq_prob model from training data
    """
    num_tags = len(tags)
    # the prob of a tag showing up in the beginning of the sentence
    init_prob = np.zeros(num_tags)
    # the prob of a tag showing up in the end of the sentence
    end_prob = np.zeros(num_tags)
    transit_prob = np.zeros((num_tags, num_tags))
    emit_prob = [{} for _ in range(num_tags)]
    freq_prob = np.zeros(num_tags)  # the prob of a tag showing up generally

    first_pair = train_data[0]
    first_word, first_tag = first_pair
    first_word = first_word.lower()
    init_prob[tags.index(first_tag)] += 1

    num_sentence = 1
    prev_tag = first_tag
    prev_word = first_word

    for word, tag in train_data[1:]:
        word = word.lower()
        if prev_word == ".":
            # the words after "." are the begining of a new sentence
            num_sentence += 1
            init_prob[tags.index(tag)] += 1

        if word == ".":
            # the words before "." are the end of a sentence
            end_prob[tags.index(prev_tag)] += 1

        transit_prob[tags.index(prev_tag)][tags.index(tag)] += 1

        if word not in emit_prob[tags.index(tag)]:
            emit_prob[tags.index(tag)][word] = 1
        else:
            emit_prob[tags.index(tag)][word] += 1

        freq_prob[tags.index(tag)] += 1

        prev_word = word
        prev_tag = tag

    # now we have each prob with counting as value, we need to switch it to prob

    # change the value of init_prob from count to percentage
    for i in range(len(init_prob)):
        count = init_prob[i]
        percentage = count / \
            num_sentence if not (
                num_sentence == 0 or count == 0) else 0.0000000001
        init_prob[i] = percentage
    normalize(init_prob)

    # change the value of end_prob from count to percentage
    for i in range(len(end_prob)):
        count = end_prob[i]
        percentage = count / \
            num_sentence if not (
                num_sentence == 0 or count == 0) else 0.0000000001
        end_prob[i] = percentage
    normalize(end_prob)

    # change the value of transit_prob from count to percentage
    for i in range(len(transit_prob)):
        total = sum(transit_prob[i])
        for j in range(len(transit_prob[i])):
            count = transit_prob[i][j]
            percentage = count / \
                total if not (total == 0 or count == 0) else 0.0000000001
            transit_prob[i][j] = percentage
        normalize(transit_prob[i])

    # change the value of emit_prob from count to percentage
    for i in range(len(emit_prob)):
        total = sum(list(emit_prob[i].values()))
        for word in emit_prob[i]:
            count = emit_prob[i][word]
            percentage = count / \
                total if not (total == 0 or count == 0) else 0.0000000001
            emit_prob[i][word] = percentage

        # normalize(emit_prob[i])
        total = sum(list(emit_prob[i].values()))
        for word in emit_prob[i]:
            p = emit_prob[i][word]
            adj_p = p / total
            if word == "to" and tags[i] == 'TO0':
                adj_p = adj_p * 100
            if word == "to" and tags[i] == 'PRP':
                adj_p = adj_p * 10
            emit_prob[i][word] = adj_p

    # change the value of freq_prob from count to percentage
    total = sum(freq_prob)
    for i in range(len(freq_prob)):
        count = freq_prob[i]
        percentage = count / \
            total if not (total == 0 or count == 0) else 0.0000000001
        freq_prob[i] = percentage
    normalize(freq_prob)

    return init_prob, end_prob, transit_prob, emit_prob, freq_prob


def handle_ambiguity(tag):  # checked
    """ return the right order of ambiguity tags, if not, return the original format"""
    if tag == "AV0-AJ0":
        return "AJ0-AV0"
    if tag == "VVN-AJ0":
        return "AJ0-VVN"
    if tag == "VVD-AJ0":
        return "AJ0-VVD"
    if tag == "NN1-AJ0":
        return "AJ0-NN1"
    if tag == "VVG-AJ0":
        return "AJ0-VVG"
    if tag == "PRP-AVP":
        return "AVP-PRP"
    if tag == "CJS-AVQ":
        return "AVQ-CJS"
    if tag == "PRP-CJS":
        return "CJS-PRP"
    if tag == "DT0-CJT":
        return "CJT-DT0"
    if tag == "PNI-CRD":
        return "CRD-PNI"
    if tag == "NP0-NN1":
        return "NN1-NP0"
    if tag == "VVB-NN1":
        return "NN1-VVB"
    if tag == "VVG-NN1":
        return "NN1-VVG"
    if tag == "VVZ-NN2":
        return "NN2-VVZ"
    if tag == "VVN-VVD":
        return "VVD-VVN"

    # if not any of the cases above, should be the right format to insert
    return tag


def read_training_list(training_files):  # checked
    """ read the training files and construct needed model"""

    if len(training_files) == 0:
        print("Not valid file")
        return

    elif len(training_files) == 1:
        res = []
        file = open(training_files[0], "r")
        for line in file.read().split('\n')[:-1]:
            pair = line.split(' : ')
            word, tag = pair[0], handle_ambiguity(pair[1])
            res.append((word, tag))

        file.close()
        return res

    else:  # more than one training files
        res = []
        for filename in training_files:
            file = open(filename, "r")
            for line in file.read().split('\n')[:-1]:
                pair = line.split(' : ')
                word, tag = pair[0], handle_ambiguity(pair[1])
                if tag == 'PUN' and tag not in punc:
                    punc.append(tag)

                res.append((word, tag))

        file.close()

        global training_size
        training_size = len(res)
        # print(training_size)
        # print(1 / training_size)
        return res


def read_test_file(filename):
    """ read the test file and store the words in test_words """
    file = open(filename, "r")
    res = []
    sentence = []
    for line in file.read().split('\n')[:-1]:
        if str(line) != '.':
            # not the end of sentence, put this word into sentence
            sentence.append(str(line))
        else:  # this is the end of the sentence
            sentence.append(str(line))
            res.append(sentence)
            sentence = []
    file.close()
    return res


def write_file(output_filename, input, results):
    """ print the test output into file """

    file = open(output_filename, "w")
    assert (len(input) == len(results))
    for i in range(len(results)):
        for j in range(len(results[i])):
            file.write(input[i][j] + " : " + results[i][j])
            file.write('\n')
    file.close()


def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    # print("Tagging the file.")

    train_data = read_training_list(training_list)
    test_data = read_test_file(test_file)
    init_prob, end_prob, transit_prob, emit_prob, freq_prob = HMM(train_data)

    result = []
    for sentence in test_data:
        # print(sentence)
        path = viterbi(
            sentence, tags, init_prob, end_prob, transit_prob, emit_prob, freq_prob)
        # print(path)
        # break
        result.append(path)
    write_file(output_file, test_data, result)


if __name__ == '__main__':
    # Run the tagger function.
    # print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    # parameters = "py tagger.py -d data/training1.txt data/training2.txt -t data/test1.txt -o Tagger/output1.txt".split()
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    print("Training files: " + str(training_list))
    print("Test file: " + test_file)
    print("Output file: " + output_file)

    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)
