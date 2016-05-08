読書メモ
=====

# 概要

Python Machine Learning本を読んでいて思ったこと、気づいたことなどをメモしていきます。

## Chapter 1

全体的にCourseraのMachine Learningの授業に出てきたような話題が扱われているっぽい。
あちらはOctaveだったので、Pythonで実装する場合にどうするのかを学習できると良いな。

## Chapter 2

Perceptronでシグモイド関数を使っていなかったり、Negativeな場合のyの値が-1だったりとCourseraの方とは微妙に違う。
-> (追記) これはneural networkのモデルによってアクティベーション関数などが違うからみたい。

あと、x_0が発火するための閾値として考えるというのは納得。なんでこの項が必要なのかがよくわかってなかったので。

NumPyを使ったことがない状態でPerceptronのコード読んでもいまいちよくわからない箇所があったので、NumPyの
チュートリアルを読む。

https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

※ 本に貼ってあったリンクは古かった様子。

ここで知ったのは

 * numpy.zerosで0初期化されたマトリックスを生成できる。
 * numpy.arrayの四則演算はエレメントワイズの演算。
 * マトリックスの積を計算するにはnumpy.dotを使う。
 * numpy.whereは第一引数のbool値がtrueなら第二引数を返して、falseなら第三引数を返す。 (https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.where.html#numpy.where)

という所。ここらが分かればPerceptronのコードは理解できた。

コードを読んでいて一瞬w_の更新がsimultaneousじゃないと思ったが、そんなことはなかった。fitメソッドの一番内側のforループは１つのexample毎にまわしているループで、w_の更新はself.w_[1:] += update * xiで行われているのだった。
ここでxiはn_features次元のfeature vectorで、updateはscalarなので、w_[1:]の各重みを同時に更新している事になる。

