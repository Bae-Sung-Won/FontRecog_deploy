import SimpleITK as sitk
import numpy as np

def command_iteration(method):
    if method.GetOptimizerIteration() == 0:
        pass
    #     print("Estimated Scales: ", method.GetOptimizerScales())
    # print(
    #     f"{method.GetOptimizerIteration():3} "
    #     + f"= {method.GetMetricValue():7.5f} "
    #     + f": {method.GetOptimizerPosition()}"
    # )

def get_similarity_transform(fixed_path, moving_path):
    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)

    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    R = sitk.ImageRegistrationMethod()

    R.SetMetricAsCorrelation()

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-8,
    )
    R.SetOptimizerScalesFromIndexShift()

    tx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Similarity2DTransform()
    )
    R.SetInitialTransform(tx)

    R.SetInterpolator(sitk.sitkLinear)

    # R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)

    # print("-------")
    # print(outTx)
    # print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    # print(f" Iteration: {R.GetOptimizerIteration()}")
    # print(f" Metric value: {R.GetMetricValue()}")

    # sitk.WriteTransform(outTx, "")

    return outTx

def get_registration(path1, path2):
    fixed = sitk.ReadImage(path1, sitk.sitkFloat32)
    moving = sitk.ReadImage(path2, sitk.sitkFloat32)

    # outTx = get_2d_rigid_transform(fixed, moving)
    outTx = get_similarity_transform(path1, path2)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    # cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
    cimg = ''
    
    # sitk.WriteImage(simg2, "test/moving.bmp")

    return simg1, simg2, cimg

def calc_mse(imageA, imageB):
    # imgA = sitk.GetArrayFromImage(imageA)
    # imgB = sitk.GetArrayFromImage(imageB)
    imgA = imageA
    imgB = imageB
    
    err = np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)
    err /= float(imgA.shape[0] * imgA.shape[1])
    
    return err

# fixed, moving, composition = get_registration("t/est/1.bmp", "test/alpha_js_0.bmp")
# print(f" MSE: {calc_mse(fixed, moving)}")