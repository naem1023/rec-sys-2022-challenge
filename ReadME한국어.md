# 전처리

```
python preprocessing.py --dataset <유저 아이템 csv파일> --output_dir <전처리 결과 저장 위치> 

예시)

python preprocessing.py --dataset RECVAE/data/train/train_ratings.csv --output_dir RECVAE/data/train/Preprocessed 

```
</br>
미리 모든 유저를 train, test 로 전처리한 결과 저장해 둠.
</br>
</br>

# 학습

```
python run.py --dataset <전처리 폴더>

예시) 

python run.py --dataset RECVAE/data/train/Preprocessed
```

</br>
모델은 자동으로 RECVAE 폴더에 valid recall@10이 높으면 model{에폭횟수}.pt 로 저장되도록 함
</br>
</br>

# 결과 파일 생성

```
python inference.py --dataset <전처리 폴더> --model <모델 위치>

예시)

python inference.py --dataset RECVAE/data/train/Preprocessed --model /home/hojin/RECVAE/model50.pt

```