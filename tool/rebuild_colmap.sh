workdir=$1
scriptdir="$( cd "$( dirname "$0" )" && pwd )"

export CUDA_VISIBLE_DEVICES=0

colmap feature_extractor \
  --database_path $workdir/colmap/database.db \
  --image_path $workdir/colmap/images  \
  --SiftExtraction.max_image_size 4096 \
  --SiftExtraction.max_num_features 16384 \
  --SiftExtraction.estimate_affine_shape 1 \
  --SiftExtraction.domain_size_pooling 1

python $scriptdir/database.py \
  --database_path $workdir/colmap/database.db \
  --txt_path $workdir/colmap/sparse_custom/cameras.txt

colmap exhaustive_matcher --database_path $workdir/colmap/database.db
mkdir -p $workdir/colmap/sparse/0

colmap point_triangulator \
  --database_path $workdir/colmap/database.db \
  --image_path $workdir/colmap/images \
  --input_path $workdir/colmap/sparse_custom \
  --output_path $workdir/colmap/sparse/0 \
  --clear_points 1

mkdir -p $workdir/colmap/dense/workspace
colmap image_undistorter \
  --image_path $workdir/colmap/images \
  --input_path $workdir/colmap/sparse/0 \
  --output_path $workdir/colmap/dense/workspace

colmap patch_match_stereo --workspace_path $workdir/colmap/dense/workspace

colmap stereo_fusion \
  --workspace_path $workdir/colmap/dense/workspace \
  --output_path $workdir/colmap/dense/workspace/fused.ply
