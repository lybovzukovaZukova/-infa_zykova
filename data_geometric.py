import Augmentor

p = Augmentor.Pipeline("dataset")
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)

p.sample(100)  # Создает 100 синтетических изображений
