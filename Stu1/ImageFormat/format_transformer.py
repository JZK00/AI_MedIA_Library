import SimpleITK as sitk


def get_series_id(data_dir):
    """
    获取系列id, 一个文件目录如果存在多个id, 则报错
    :param data_dir: 存放dicom文件的目录
    :return: 1个系列id
    """
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_dir)
    assert len(series_id) == 1, "存在不同系列id的dicom数据"
    return series_id[0]


def run_transform(dicom_dir, save_path):
    """
    转换主程序
    :param dicom_dir: 存放dicom文件的目录
    :param save_path: 转成  .nii 或 .nii.gz 后保存的地址, 例如 './trans.nii' 或 './trans.nii.gz'
    """
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir, get_series_id(dicom_dir))
    series_reader = sitk.ImageSeriesReader()  # 读取数据端口
    series_reader.SetFileNames(series_file_names)
    images = series_reader.Execute()  # 读取数据
    sitk.WriteImage(images, save_path)


if __name__ == '__main__':
    # 存放dicom文件的目录
    dicom_dir_ = r"../../datasets/demo_DICOM/emphysema"

    # 转成 NIFTI 后的保存路径， 后缀名为 .nii 或 .nii.gz 都行
    save_path_ = r"../../datasets/demo_DICOM/emphysema.nii"

    run_transform(dicom_dir_, save_path_)

    print('转换完成')
