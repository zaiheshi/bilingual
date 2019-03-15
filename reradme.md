要处理的文件是src与mt
1. 把文件切分为多个子文件(用到codecs)
2. 对应的文件src_、mt_转换为[Example object、Example object、Example object、、], 其中
Example object包含类变量"src"、"tgt"、"indices", "src"/"tgt"为某行的word列表,
"indices"为该行的索引号, 最终保存为多个pt文件(用到glob、torchtext.data.Dataset的
继承类Dataset, 主要参数为(Example object list, [("src",fields["src"]), ("tgt",fields["tgt"]),
("indices",fields["indices"])), 保存为pt文件时把第二项置空, 为什么不保存在一起?
)

fields的作用？Eaxmple object list如何获得的？使用torchtext是否真的简化了操作? 重点
在Dataset类，以及生成器的使用, generator的使用在逻辑上简化了Eaxmple object list,
generator的代码需要学会

未使用torchtext时的代码
dic = {"word1": 9, "word2": 10, "word3": 11, "word4":99, "word5": 3000}
wrd = "word5 word4 word3 word2 word1".split()
word2idx = dict()
for word in dic:
    if word not in word2idx:
        word2idx[word] = len(word2idx)
idx2word = {v: k for k, v in word2idx.items()}
idx_seq = [word2idx.get(word) for word in wrd]
print(word2idx)
print(idx2word)
print(idx_seq)

对字典按key排序lst = sort(dict.items(), key = lamada x: x[0])


3. 根据得到的pt文件构建vocab, 通过Counter()得到字典
{"src":包含所有src word以及数量的Counter对象,
"tgt":包含所有tgt word以及数量的Counter对象,
"indices": 空Counter()}, 将word与index的映射关系记录在fields[*].vocab中，
最终把[(*, vocab)]保存为pt文件， *为src/tgt