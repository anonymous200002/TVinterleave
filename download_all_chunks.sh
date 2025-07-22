#!/bin/bash
mkdir -p ymju_interleve_zip
cd ymju_interleve_zip

# Due to bandwidth and hosting constraints, only chunk_0_40.tar.gz is currently available for public download. If you are interested in accessing the full dataset (6.5TB), please contact the authors directly (18621395721@163.com).

base_url="https://flagchat.ks3-cn-beijing.ksyun.com/runway_log/ymju_interleve_zip"

files=(
  chunk_0_40.tar.gz
  chunk_40_80.tar.gz
  chunk_80_120.tar.gz
  chunk_120_160.tar.gz
  chunk_160_200.tar.gz
  chunk_200_240.tar.gz
  chunk_240_280.tar.gz
  chunk_280_320.tar.gz
  chunk_320_360.tar.gz
  chunk_360_400.tar.gz
  chunk_400_440.tar.gz
  chunk_440_480.tar.gz
  chunk_480_520.tar.gz
  chunk_520_560.tar.gz
  chunk_560_600.tar.gz
  chunk_600_640.tar.gz
  chunk_640_680.tar.gz
  chunk_680_720.tar.gz
  chunk_720_760.tar.gz
  chunk_760_800.tar.gz
  chunk_800_840.tar.gz
  chunk_840_880.tar.gz
  chunk_880_920.tar.gz
  chunk_920_960.tar.gz
  chunk_960_1000.tar.gz
  chunk_1000_1018.tar.gz
)

for file in "${files[@]}"; do
  wget "${base_url}/${file}"
done
