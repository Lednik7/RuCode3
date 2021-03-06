# Всероссийский учебный фестиваль по искусственному интеллекту и программированию [RuCode 3.0](https://rucode.net/)
Его организаторами стали 15 ведущих вузов и общественных организаций со всей России. Фестиваль поддержало Министерство науки и высшего образования Российской Федерации, а индустриальными партнерами мероприятия стали «МегаФон», «Яндекс», «Сбер» и «Газпромбанк».

## :tada: катбуст: скоростной мангуст - 2 место :tada:
![image](https://user-images.githubusercontent.com/52196169/119946596-77782580-bf9f-11eb-9cb6-6753d6f9d66c.png)

[Ссылка на соревнование и результаты](https://www.kaggle.com/c/anime-recommendation-rucode/leaderboard)

[Ссылка на трансляцию с выступлениями всех участников](https://clck.ru/V8dJn)

## Задача
На основе общей информации об аниме и прошлых оценок пользователя нужно будет научиться понимать, какие аниме, которые пользователь еще не смотрел, понравятся этому пользователю.
По паре (пользователь, анимэ) нужно предсказать, насколько данному пользователю понравится данное аниме по шкале от 1 до 10.

## Решение
Используем библиотеку коллаборативной фильтрации [tf-recsys](https://github.com/WindQAQ/tf-recsys) и алгоритм [SVD++](https://en.wikipedia.org/wiki/Singular_value_decomposition), обучаем на GPU. В ходе работы применяли разные модели: catboost, SVD, Annoy.

Наши решения можно найти в папке [notebooks](https://github.com/Lednik7/RuCode3/tree/main/notebooks), а презентацию в [presentation](https://github.com/Lednik7/RuCode3/blob/main/presentation/RuCode3_v1.8.pptx.pdf).

## Contributors

[@Арсений Шахматов](https://github.com/cene555) |
[@Степан Шабалин](https://github.com/neverix) |
[@Максим Герасимов](https://github.com/Lednik7)

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Lednik7/RuCode3/blob/main/LICENSE) file for details
