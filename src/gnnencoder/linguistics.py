import numpy as np
import torch
import stanza
import re

# 下载中文简体的StandfordNLP，默认在~/stanza-resources
# stanza.download('zh-hans')


class SentenceAnalyzer:
    def __init__(self, top_k=0.3):
        self.top_k = top_k
        # 需要关闭自动下载、开启预分词、开启 GPU
        if torch.cuda.is_available():
            self.nlp = stanza.Pipeline('zh-hans', processors='tokenize,pos,lemma,depparse', use_gpu=True, download_method=None, tokenize_pretokenized=True)
        else:
            self.nlp = stanza.Pipeline('zh-hans', processors='tokenize,pos,lemma,depparse', use_gpu=False, download_method=None, tokenize_pretokenized=True)
        self.pos2id = {}
        self.dep2id = {}


    def _get_unique_tags_deps(self, sentences):
        all_tags = set()
        all_deps = set()

        for sentence in sentences:
            doc = self.nlp([list(sentence)])
            for word in doc.sentences[0].words:
                all_tags.add(word.upos)
                all_deps.add(word.deprel)
        # Create all possible combinations of pos tags
        all_combinations = {f"{tag1}-{tag2}" for tag1 in all_tags for tag2 in all_tags}

        self.pos2id = {tag: i for i, tag in enumerate(all_combinations)}
        self.dep2id = {dep: i for i, dep in enumerate(all_deps)}

    def pos_tagging(self, sentence):
        # # note 预先分词，保持统一的方法
        doc = self.nlp([list(sentence)])
        n = len(doc.sentences[0].words)
        pos_matrix = np.zeros((n, n), dtype=int)

        for i, word_i in enumerate(doc.sentences[0].words):
            for j, word_j in enumerate(doc.sentences[0].words):
                # print(f"{word_i.upos}-{word_j.upos}")
                pos_matrix[i, j] = self.pos2id[f"{word_i.upos}-{word_j.upos}"]

        return pos_matrix

    def dependency_parsing(self, sentence):
        doc = self.nlp([list(sentence)])
        n = len(doc.sentences[0].words)
        dep_matrix = np.zeros((n, n), dtype=int)

        for word in doc.sentences[0].words:
            head_index = word.head - 1
            if head_index >= 0:  # Check to avoid index -1 for the root
                dep_matrix[word.id - 1, head_index] = self.dep2id[word.deprel]

        return dep_matrix

    def batch_encode_sentences(self, sentences):
        self._get_unique_tags_deps(sentences)

        s = len(sentences)
        max_n = max([len(self.nlp([list(sentence)]).sentences[0].words) for sentence in sentences])

        result = np.zeros((s, max_n, max_n, 2), dtype=int)

        for idx, sentence in enumerate(sentences):
            pos_res = self.pos_tagging(sentence)
            dep_res = self.dependency_parsing(sentence)
            result[idx, :pos_res.shape[0], :pos_res.shape[1], 0] = pos_res
            result[idx, :dep_res.shape[0], :dep_res.shape[1], 1] = dep_res

        return result

    def parse_output_for_center_words(self, sentence, output):
        """
        解析输出
        """
        output_text = re.findall(r'\[(.*?)\]', output)
        if output_text:
            # 使用逗号分割整体词组
            output_list_first = output_text[0].split(', ')
            # 处理每个词组，去掉序号和制表符，并拆分成多个字符串
            output_list_second = []
            for entry in output_list_first:
                # 去除可能的序号（如1.、2.）和空格
                cleaned_entry = re.sub(r'\d+\.', '', entry).strip()
                # 如果存在\t，则分割为多个字符串
                if '\t' in cleaned_entry:
                    split_entry = cleaned_entry.split('\t')
                    output_list_second.extend(split_entry)  # 扩展到最终列表中
                else:
                    output_list_second.append(cleaned_entry)  # 直接添加到最终列表中
            # 检查是否存在于句子中，如果存在则添加到最终列表中
            output_list_third = []
            for phrase in output_list_second:
                if phrase in sentence:
                    output_list_third.append(phrase)
                else:
                    new_str = ''
                    for char in phrase:
                        if char in sentence:
                            new_str += char
                        else:
                            continue
                    if new_str:
                        output_list_third.append(new_str)
                    else:
                        continue
            return output_list_third
        else:
            raise ValueError("The output string is not in the correct format!")

    def get_center_word(self, sentences, center_words):
        # 找出 center word 在句子中的索引，注意可能是词组
        center_word_indices = []

        for i, sentence in enumerate(sentences):
            sentence_indices = []
            for words_group in center_words[i]:
                for word in words_group:
                    if len(list(word)) > 1:
                        sentence_words, word_tokens = list(sentence), list(word)
                        indices = [sentence_words.index(token) for token in word_tokens if token in sentence_words]
                    else:
                        # 使用 split() 获取单词的索引
                        indices = [list(sentence).index(word)]
                    sentence_indices.append(indices)
            center_word_indices.append(sentence_indices)
            # print(center_word_indices)

        sentences_features = self.batch_encode_sentences(sentences)
        # print(sentences_features)
        
        for i, center_word_index in enumerate(center_word_indices):
            for j, word_index in enumerate(center_word_index):
                # 如果有中心词包含多个 token
                if len(word_index) > 1:
                    max_value, max_idx = -1, None
                    # 取句法非零关系最多的 token 作为中心词
                    for idx in word_index:
                        total_count = np.count_nonzero(sentences_features[i][idx, :, 1]) + np.count_nonzero(
                            sentences_features[i][:, idx, 1]) - (1 if sentences_features[i][idx, idx, 1] != 0 else 0)
                        if total_count > max_value:
                            max_value = total_count
                            max_idx = idx
                    center_word_indices[i][j] = max_idx

        return sentences_features, center_word_indices

    def selected_top_k_similarity(self, similarity):
        # 对相似度矩阵的每行进行降序排序，选择其中的前k%置1.目的是找出一个batch中相似度最高的一些句子
        similarity_matrix = np.zeros_like(similarity)
        for i in range(len(similarity)):
            row = similarity[i]
            sorted_indices = np.argsort(row)[::-1]
            top_k_num = int(len(similarity) * self.top_k)
            for j in range(top_k_num):
                idx = sorted_indices[j]
                similarity_matrix[i][idx] = 1
        return similarity_matrix

    def linguistic_feature(self, sentences, outputs):
        '''

        :param sentences: original sentences in input_text
        :param outputs: right options in output_text
        :return:
        '''
        if len(sentences) != len(outputs):
            AssertionError('The length of sentences and outputs must be the same')

        center_words = [self.parse_output_for_center_words(sentence, output) for sentence, output in zip(sentences, outputs)]
        # print(f"center_words - {center_words}")
        sentences_features, center_word_indices = self.get_center_word(sentences, center_words)
        similarity = calculate_similarity(sentences_features, center_word_indices)

        # 打印相似度，统计后再确定阈值
        # print(similarity.mean(), similarity.max())
        # print(similarity)

        # 生成语言特征 (0-1 矩阵)
        linguistic_feature = self.selected_top_k_similarity(similarity)

        return linguistic_feature

def gauss_weight(center_idx, length, sigma=2.0):
    """
    Calculate the weight of each token based on its distance to the center token.
    """
    weights = np.zeros(length)
    for idx in center_idx:
        dists = np.abs(np.arange(length) - idx)
        current_weights = np.exp(-np.square(dists) / (2 * sigma ** 2))
        weights = np.maximum(weights, current_weights)
    return weights


def hamming_distance(T1, T2):
    """
    Calculate the hamming distance between two feature tensors.
    """
    return np.sum(T1 != T2)


def sigmoid(distance, scale=0.01):
    """
    Use sigmoid to calculate similarity score based on distance.
    scale 越大，函数会更陡，即对输入的微小变化反应会更敏感。
    """
    return 1 / (1 + np.exp(scale * distance))

def calculate_similarity(matrix, center_tokens):
    s, n, _, _ = matrix.shape
    similarity_matrix = np.zeros((s, s))

    # Iterating over each sentence pair
    for i in range(s):
        for j in range(s):
            if i >= j:
                continue

            center_i = center_tokens[i]
            center_j = center_tokens[j]

            weights_i = gauss_weight(center_i, n)
            weights_j = gauss_weight(center_j, n)

            weighted_matrix_i = matrix[i] * weights_i[:, np.newaxis, np.newaxis]
            weighted_matrix_j = matrix[j] * weights_j[:, np.newaxis, np.newaxis]

            min_centers = min(len(center_i), len(center_j))
            distances = []
            for c_i in center_i[:min_centers]:
                for c_j in center_j[:min_centers]:
                    dist = hamming_distance(weighted_matrix_i[c_i], weighted_matrix_j[c_j])
                    distances.append(dist)

            # We average the similarity score over the minimal center tokens
            similarity_matrix[i][j] = similarity_matrix[j][i] = np.mean([sigmoid(dist) for dist in distances])

    return similarity_matrix


if __name__ == "__main__":
    # 示例
    inputs = [
        "你就是空气 让我能呼吸 像四季自在交替 我愿自己是暖暖的毛衣 可以给你",
        "其实班长的故事一箩筐 当年也想过要走天涯去流浪 只是穿上这身军装咱就不太一样  男人就要把山扛在肩上  小张",
        "要爱你赖在嘴边 别想在轻易善变 就像你离去时的疯癫 嘟嘟 你就那山顶的雾",
        "春风一度葬良宵 匆匆流年忘今朝 岁月如同一把刀 难忘那些雪花飘 我漫步在金沙滩",
        "你将我磨成利器 恋爱路有幸捱不死 竞技场上当嬉戏 今天讲来彷佛一世纪",
        "月是你 是你命中一曲不沉的舟 风悠悠云悠悠 风悠悠云悠悠",
        "这就是艾晚。 她出示证件的动作犹如电光石火，完全不把看家护院的大兵放在眼里。 万良感到被人轻视的愤慨。",
        "变成翅膀守护你 你要相信 相信我们会像童话故事里 幸福和快乐是结局 你哭着对我说"
    ]
    outputs = [
        "B: [你, 空气, 必需的属性]",
        "C: [责任, 山, 沉重的感觉]",
        "D: [你, 山顶的雾, 多变的状态]",
        "D: [岁月, 刀, 伤人的能力]",
        "B: [我, 利器, 尖锐的外表]",
        "B: [月, 你命中一曲不沉的舟, 飘荡的事物]",
        "C: [动作, 电光火石, 迅速的动作]",
        "A: [我们的故事, 童话故事, 美满的结局]"
    ]

    analyzer = SentenceAnalyzer(top_k=0.5)

    linguistic_feature = analyzer.linguistic_feature(inputs, outputs)
    print(linguistic_feature)