workdir=$1
export CUDA_VISIBLE_DEVICES=0

# 清理之前生成的中间文件和文件夹
rm -rf $workdir/colmap

# 准备必要的文件夹结构
mkdir -p $workdir/colmap/images

# 假设你的图片已经在 $workdir/images 文件夹下，将其复制到 colmap/images 文件夹下
cp -r $workdir/frames/0000/* $workdir/colmap/images/

# 特征提取
colmap feature_extractor \
    --database_path $workdir/colmap/database.db \
    --image_path $workdir/colmap/images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.max_image_size 3200 \
    --SiftExtraction.max_num_features 16384 \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1

# 检查特征提取是否成功
if [ $? -ne 0 ]; then
    echo "Feature extraction failed"
    exit 1
fi

# 特征匹配
colmap exhaustive_matcher \
    --database_path $workdir/colmap/database.db

# 检查特征匹配是否成功
if [ $? -ne 0 ]; then
    echo "Exhaustive matching failed"
    exit 1
fi

# 三角化
mkdir -p $workdir/colmap/sparse/0
colmap mapper \
    --database_path $workdir/colmap/database.db \
    --image_path $workdir/colmap/images \
    --output_path $workdir/colmap/sparse \
    --Mapper.ba_global_function_tolerance 0.000001

# 检查三角化是否成功
if [ $? -ne 0 ]; then
    echo "Mapping failed"
    exit 1
fi

# 图像去畸变
mkdir -p $workdir/colmap/dense/workspace
colmap image_undistorter \
    --image_path $workdir/colmap/images \
    --input_path $workdir/colmap/sparse/0 \
    --output_path $workdir/colmap/dense/workspace

# 检查图像去畸变是否成功
if [ $? -ne 0 ]; then
    echo "Image undistortion failed"
    exit 1
fi

#exit 0

# PatchMatch 立体匹配
colmap patch_match_stereo \
    --workspace_path $workdir/colmap/dense/workspace

# 检查PatchMatch是否成功
if [ $? -ne 0 ];then
    echo "PatchMatch stereo failed"
    exit 1
fi

# 立体融合
colmap stereo_fusion \
    --workspace_path $workdir/colmap/dense/workspace \
    --output_path $workdir/colmap/dense/workspace/fused.ply

# 检查立体融合是否成功
if [ $? -ne 0 ]; then
    echo "Stereo fusion failed"
    exit 1
fi

echo "COLMAP pipeline completed successfully."
