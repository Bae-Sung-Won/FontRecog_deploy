import streamlit as st
from apps.trace_skeleton import Trace
from apps.fix import fix
from apps.registration import get_registration
import SimpleITK as sitk
from apps.model import autoencoders
from apps.metrics import metrics
from collections import Counter
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
import shutil
import time
import numpy as np
import natsort
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from copy import deepcopy
from stqdm import stqdm
import warnings 
warnings.filterwarnings('ignore')

print(torch.__version__)

def save_upload_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)
    if os.path.splitext(file.name)[1] == '.bmp':
        with open(os.path.join(directory, file.name), 'wb') as f:
            f.write(file.getbuffer())
    else:
        with open(os.path.join(directory, os.path.splitext(file.name)[0] + '.bmp'), 'wb') as f:
            f.write(file.getbuffer())

def s1(paths):
    # trace_skeleton.py module load   
    im0, _ = Trace(paths)                          ### Trace 첫번째 인자는 이미지에 trace_skeleton 적용, 두번째 인자는 trace_skeleton 부분만 추출
    # 원본 이미지 불러오기
    im = np.fromfile(paths, np.uint8)              ### 이미지 데이터에 한글 경로가 있을 경우, np.fromfile함수를 통해 바이너리 데이터를 넘파이 행렬로
    im = cv2.imdecode(im, cv2.IMREAD_COLOR)        ### cv2.imdecode를 통해 복호화 과정을 거쳐서 opencv에서 이용할 수 있는 형태로 변환
    # 원본 이미지에 hsv로 빨간색만 필터링
    im = fix(im).hsv()
    sang_im = deepcopy(im)
    sang_im = cv2.cvtColor(sang_im, cv2.COLOR_BGR2GRAY)
    sang_im = cv2.threshold(sang_im,127,255, cv2.THRESH_BINARY)[1]
    sang_im = cv2.cvtColor(sang_im, cv2.COLOR_GRAY2BGR)
    # findContours까지 적용한 이미지 그리고 copy
    im = fix(im).Contour()
    # hsv -> BGR -> GRAY -> threshold
    im = fix(im).threshold()
    # 노이즈 제거 및 선명화 작업
    im = fix(im).Noise()
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) # 추가(3channel로 변경)
    return sang_im, im, im0                   # sang_im은 생짜 이미지,  im은 전처리 이미지, im0은 trace_skeleton 이미지를 return

def save(file, name):
    path = os.path.join(os.getcwd(), 'dataset', 'regist', name)
    type = os.path.splitext(path)[1]
    result, encoded_img = cv2.imencode(type, file)
    if result:
        with open(path, mode='w+b') as f:
            encoded_img.tofile(f)

# 나눔고딕 폰트 추가 함수
@st.cache_data
def fontRegistered():
    font_dirs = os.getcwd() + '/customFonts'
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    fm.fontManager.addfont(font_files[0])
    fm._load_fontmanager(try_read_cache=False)

# ------------------------------------------------------------------------------------------------------------------
def app():
    st.markdown("""
    <style>
    .small-font {
        font-size:18px !important;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: black;'>Font Classification</h1>", unsafe_allow_html=True)
    st.subheader(":blue[When restarting, be sure to press F5.] :sunglasses:", divider='rainbow')
    
    st.write('한글 이름의 이미지 파일을 업로드 하세요. Ex) 가.bmp, 가.png ...')
    image = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'bmp', 'jpg', 'jpeg'])

    with st.sidebar:
        st.markdown('<p style=color:black; "small-font">' +'Inference time 평균 4~5분 정도 소요'+ '</p>', unsafe_allow_html=True)
        st.write('available device: ', torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        st.subheader('csv 파일을 다운 받으면, 이미지 간 비교', divider='rainbow')
        st.markdown('<p style=color:red; "small-font">' +'1. MSE는 0에 가까울수록 유사'+ '</p>', unsafe_allow_html=True)
        st.markdown('<p style=color:blue; "small-font">' +'2. Cos는 1에 가까울수록 유사'+ '</p>', unsafe_allow_html=True)
        st.markdown('<p style=color:green; "small-font">' +'3. SSIM는 1에 가까울수록 유사'+ '</p>', unsafe_allow_html=True)


    # csv download button 누르면서, True가 반환되면서 다시 재시작을 방지하기 위한 callback 함수 설정.
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = True

    if (image is not None) and (st.session_state.button_clicked):
        left_co, cent_co, last_co = st.columns(3)
        with cent_co:
            st.image(image, width=300, caption= 'input data')
        # st.markdown('<p class="small-font">' +'input data'+ '</p>', unsafe_allow_html=True)

        # 업로드 파일 저장(확장명을 모두 bmp파일로 변환. -> registration하기 위해서)
        save_upload_file('dataset', image)   
        
        # clearn 이미지, 저장한 이미지 경로 load
        recon_data = os.path.join(os.getcwd(), 'dataset', 'clearn_image')
        recon_data_numbers = natsort.natsorted(os.listdir(recon_data))
        image = natsort.natsorted(os.listdir(os.path.join(os.getcwd(), 'dataset')))[-1]
        image = os.path.join(os.getcwd(), 'dataset', image)

        # 전역변수 설정.
        st.session_state.recon_data = recon_data
        st.session_state.image = image

        # 1. 이미지 resize 적용
        def process_image(path):
            img = np.fromfile(path, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            return img

        def center_crop(img, set_size1, set_size2):
            h, w, c = img.shape
            crop_width = set_size1
            crop_height = set_size2
            mid_x, mid_y = w//2, h//2
            offset_x, offset_y = crop_width//2, crop_height//2
            crop_img = img[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
            return crop_img
        
        type = os.path.splitext(image)[1]
        result, encoded_img = cv2.imencode(type, center_crop(process_image(image), 400, 300))
        if result:
            with open(image, mode='w+b') as f:                
                encoded_img.tofile(f)


        # 2. 이미지 trace_skeleton 적용
        _, _, im = s1(image)
        type = os.path.splitext(image)[1]
        result, encoded_img = cv2.imencode(type, im)
        if result:
            with open(image, mode='w+b') as f:                
                encoded_img.tofile(f)

        left_co, cent_co, last_co = st.columns(3)
        with cent_co:
            st.image(image, width=300, caption= 'preprocessing data')

        # 3. 12개의 class에 존재하는 해당 글자와 모두 Registration을 진행 후, 레지스트 폴더에 저장
        if not os.path.exists(os.path.join('dataset', 'regist')):
            os.makedirs(os.path.join('dataset', 'regist'))
            for i in stqdm(recon_data_numbers, desc='In registering ...'):
                location = os.listdir(os.path.join(recon_data, i))
                if os.path.basename(image) in location:
                    paths = os.path.join(recon_data, i, os.path.basename(image))
                    path = image
                    _, moving, _ = get_registration(paths, path)           # # fixed를 기준으로, moving 이미지를 변환
                    moving = sitk.GetArrayFromImage(moving)
                    moving = np.reshape(moving, (moving.shape[0], moving.shape[1], 1))
                    moving = cv2.cvtColor(moving, cv2.COLOR_GRAY2BGR)
                    moving = cv2.cvtColor(moving, cv2.COLOR_BGR2RGB)
                    save(moving, i+os.path.splitext(image)[1])

            # 4. 오토인코더 모델을 통한 이미지 복원 진행.
            # left_co, cent_co, last_co = st.columns(3)
            # with cent_co:
            with st.spinner('AI Reconsturction ...'):
                # 4.1 모델 load
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                print(f'Selected device: {device}')
                model = autoencoders()
                model.to(device)
                model.load_state_dict(torch.load('./model/autoencoder.pt', map_location=device))

                # 4.2 복원 진행 후, 레지스트 폴더에 저장
                transform = transforms.Compose([transforms.ToTensor()])      # tensor 형태로 변환.
                location = natsort.natsorted(os.listdir(os.path.join(os.getcwd(), 'dataset', 'regist'))) # n개의 registration 적용 이미지 경로.

                class CustomDataset(Dataset):                                # Dataset으로 묶기.
                    def __init__(self, input_imgs):
                        self.input_imgs = input_imgs
                    def __len__(self):
                        return len(self.input_imgs)
                    def __getitem__(self, idx):
                        input_img = self.input_imgs[idx]
                        return input_img

                # Data Loader 준비.
                test = []
                for i in location:
                    test.append(transform(process_image(os.path.join(os.getcwd(), 'dataset', 'regist') + '/' + i)))            
                test = CustomDataset(test)
                if torch.cuda.is_available():
                    test_loader = DataLoader(test, batch_size=16, shuffle=False)                       # gpu면, batch 16
                else:
                    test_loader = DataLoader(test, batch_size=1, shuffle=False)                       # cpu면, batch 1

                # reconstruction image 생성.
                model.eval()
                with torch.no_grad():
                    x_hat = []  # test_loader pred
                    for index, image_batch in stqdm(enumerate(test_loader), desc='model running ...'):
                        image_batch = image_batch.to(device)
                        pred = model(image_batch)
                        x_hat += [pred]
        
                        if index == 0:
                            print(pred.shape)

                def sunmyung(img):                        # 일정 픽셀값 이상 전부 255로 변환(선명화)
                    img = img[0]                          # 3,300,400 -> 300,400
                    a = []
                    for i in img:
                        b = []
                        for j in i:
                            if j < torch.tensor(0.3):
                                b.append(j)
                            elif j >= torch.tensor(0.3) and j <= torch.tensor(1.):
                                b.append(torch.tensor(1.))
                        a.append(b)
                    a = torch.tensor(a).repeat(3, 1, 1)   # 300,400 -> 3,300,400
                    return a

                # 저장하기.
                tf_toPILImage=transforms.ToPILImage()     
                index = -1
                for i in x_hat:
                    for j in i:
                        index += 1
                        j = sunmyung(j.detach().cpu())
                        name = location[index]
                        tf_toPILImage(j).save(os.path.join(os.getcwd(), 'dataset', 'regist', name))

        st.success('Reconstruction Complete!', icon="✅")


        # 5. 성능 평가.
        # left_co, cent_co, last_co = st.columns(3)
        # with cent_co:
        with st.spinner('Performance evaluation in progress ...'):
            location = natsort.natsorted(os.listdir(os.path.join(os.getcwd(), 'dataset', 'regist'))) # n개의 registration 적용 이미지 경로.
            a1 = []
            a2 = []
            a3 = []
            for i in stqdm(location, desc='This is slow task'):
                mse = []
                cos = []
                ss = []
                for j in location:
                    paths = os.path.join(recon_data, os.path.splitext(j)[0], os.path.basename(image))
                    path = os.path.join(os.getcwd(), 'dataset', 'regist', i)
                    time.sleep(0.5)
                    print(path)
                    fixed, moving, _ = get_registration(paths, path)           # # fixed를 기준으로, moving 이미지를 변환
                    fixed, moving = sitk.GetArrayFromImage(fixed), sitk.GetArrayFromImage(moving)
                    mse_top, cos_top, ss_top = metrics(fixed, moving).rank()
                    mse.append((os.path.splitext(j)[0] +'_'+ os.path.basename(image), mse_top))
                    cos.append((os.path.splitext(j)[0] +'_'+ os.path.basename(image), cos_top))
                    ss.append((os.path.splitext(j)[0] +'_'+ os.path.basename(image), ss_top))
                mse = {z1 : z2 for z1, z2 in mse}
                cos = {z1 : z2 for z1, z2 in cos}
                ss = {z1 : z2 for z1, z2 in ss}
                a1.append(mse)
                a2.append(cos)
                a3.append(ss)

            best_score1 = np.inf
            best_score2 = 0
            best_score3 = 0
            for index, (i, j, k) in enumerate(zip(a1, a2, a3)):
                index += 1
                if best_score1 > sorted(i.items(), key=lambda x:x[1], reverse=False)[0][1]:
                    zz1 = []              # mse
                    zz1.append((index, i))
                    best_score1 = sorted(i.items(), key=lambda x:x[1], reverse=False)[0][1]

                if best_score2 < sorted(j.items(), key=lambda x:x[1], reverse=True)[0][1]:
                    zz2 = []              # cos
                    zz2.append((index, j))
                    best_score2 = sorted(j.items(), key=lambda x:x[1], reverse=True)[0][1]

                if best_score3 < sorted(k.items(), key=lambda x:x[1], reverse=True)[0][1]:
                    zz3 = []              # ssim
                    zz3.append((index, k))
                    best_score3  = sorted(k.items(), key=lambda x:x[1], reverse=True)[0][1]


            if len(set(Counter([zz1[0][0], zz2[0][0], zz3[0][0]]).values())) == 1:
                zzz = zz1[0][0]
                zz1 = []
                zz1.append((zzz, a1[zzz - 1]))
                zz2 = []
                zz2.append((zzz, a2[zzz - 1]))
                zz3 = []
                zz3.append((zzz, a3[zzz - 1]))
            else:
                zzz = sorted(Counter([zz1[0][0], zz2[0][0], zz3[0][0]]).items(), key=lambda x: x[1], reverse=True)[0][0]
                zz1 = []
                zz1.append((zzz, a1[zzz - 1]))
                zz2 = []
                zz2.append((zzz, a2[zzz - 1]))
                zz3 = []
                zz3.append((zzz, a3[zzz - 1]))

        # 6. 저장 및 확인하기.
        # 클래스 번호 -> 이름으로 변경해주기.
        classes = [os.path.splitext(i)[0][:-2] for i in zz1[0][1].keys()]
        names = ['아트방-고딕체', '아트방-명조체', '아트방-굴림체', '아트방-솔체', '아트방-전서체',
                 '알파-명조체', '알파-굴림체', '알파-솔체', '알파-고딕체', '알파-전서체',
                 '파란불-명조체', '파란불-전서체'
                 ]
        names = [names[int(i)-1] for i in classes]
        df = pd.DataFrame({'클래스':classes, '클래스_이름': names, 'mse': zz1[0][1].values(), 'cos':zz2[0][1].values(), 'ssim':zz3[0][1].values()}).set_index('클래스')
        # st.write('result')
        st.markdown("<p style='text-align: center; color: red; font-size:180%'>Result </p>", unsafe_allow_html=True)
        st.table(df)

        tmp_df = deepcopy(df.reset_index())
        if tmp_df not in st.session_state:
            st.session_state.df = tmp_df

        # 차트 그리기.
        # plt.rcParams['font.family'] ='Malgun Gothic'
        # plt.rcParams['axes.unicode_minus'] =False
        
        # 나눔고딕 폰트 load.
        fontRegistered()
        plt.rc('font', family='NanumGothic')
        fig, ax = plt.subplots(1, 3, figsize=(18,8))
        palette = sns.color_palette("bright", 12)

        plt.subplot(1, 3, 1)
        sns.barplot(data=df.reset_index(), x='클래스', y='mse',  palette=palette, dodge=False)
        plt.title('MSE')

        plt.subplot(1, 3, 2)
        sns.barplot(data=df.reset_index(), x='클래스', y='cos', palette=palette, dodge=False)
        plt.title('Cosine Similarity')

        plt.subplot(1, 3, 3)
        sns.barplot(data=df.reset_index(), x='클래스', y='ssim', hue='클래스_이름', palette=palette, ax=ax[2], dodge=False)
        box = ax[2].get_position()
        ax[2].set_position([box.x0, box.y0, box.width * 1, box.height])
        ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.8))
        plt.title('SSIM')

        st.pyplot(fig)


        left_co, last_co = st.columns(2)
        with left_co:
            st.session_state.best_name = str(zz1[0][0]) + '.bmp'
            st.image(os.path.join(os.getcwd(), 'dataset', 'regist', st.session_state.best_name), width=300, caption= 'Best Reconstruction data')

        with last_co:
            st.session_state.best_number = os.path.splitext(sorted(zz1[0][1].items(), key=lambda x:x[1], reverse=False)[0][0])[0][:-2]
            st.image(os.path.join(recon_data, st.session_state.best_number ,os.path.basename(image)), width=300, caption= 'Best Similar data :' + df['클래스_이름'][int(st.session_state.best_number) - 1])
            

        # 다운 버튼 누르면, 다시 시작하는 문제 발생하기 때문에, callback 함수 설정해서, 다운 버튼 누를 시, callback함수 실행하여, False값 반환.
        # streamlit 에서 제공되는 기능들은 button, checkbox등 모두 누르기 전엔 Flase, 누른 후에는 True값을 반환한다.
        def callback():
            #Button was clicked!
            st.session_state.button_clicked = False
            
        @st.cache_data
        def convert_df(df):
            return df.to_csv()
        csv = convert_df(df)

        if st.download_button(
            label="CSV Download",
            data=csv,
            file_name='result.csv',
            mime='text/csv',
            on_click=callback
        ):
            st.write('Thanks for downloading!')
            time.sleep(2)

    # df값 유지 안되니깐, session_state로 df저장해서 사용해주기.(금요일날 추가해서 돌리기.)
    
    # 2번째 페이지 만들지 말고, 걍 페이지 자동 restart 해서, 이미지 띄우고 결과 보여주기.
    elif not (st.session_state.button_clicked):
        left_co, cent_co, last_co = st.columns(3)

        with left_co:
            st.markdown("<p style='text-align: center; color: red; font-size:120%'> Reconstruction data </p>", unsafe_allow_html=True)
            for i in stqdm(st.session_state.df['클래스']):
                st.markdown("<p style='text-align: center; font-size:90%'> </p>", unsafe_allow_html=True)
                st.image(os.path.join(os.getcwd(), 'dataset', 'regist', st.session_state.best_name), width=225, caption= 'Reconstruction data')

        with cent_co:
            st.markdown("<p style='text-align: center; color: red; font-size:120%'> Performance </p>", unsafe_allow_html=True)
            st.markdown("""<style> [data-testid="stMetricValue"] {font-size: 15px;} </style>""", unsafe_allow_html=True)

            for i in stqdm(st.session_state.df['클래스']):
                if str(i) == st.session_state.best_number:
                    st.metric(label='_', value = 'MSE', delta='+' + str(st.session_state.df[st.session_state.df['클래스'] == i]['mse'].values[0]), label_visibility='collapsed')
                    st.metric(label='_', value = 'Cosine Similarity', delta='+' + str(st.session_state.df[st.session_state.df['클래스'] == i]['cos'].values[0]), label_visibility='collapsed')
                    st.metric(label='_', value = 'SSIM', delta='+' + str(st.session_state.df[st.session_state.df['클래스'] == i]['ssim'].values[0]), label_visibility='collapsed')
                    st.markdown("<p style='text-align: center; font-size:10%'>. </p>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; font-size:10%'>. </p>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; font-size:10%'>. </p>", unsafe_allow_html=True)


                else:
                    st.metric(label='_', value = 'MSE', delta='-' + str(st.session_state.df[st.session_state.df['클래스'] == i]['mse'].values[0]), label_visibility='collapsed')
                    st.metric(label='_', value = 'Cosine Similarity', delta='-' + str(st.session_state.df[st.session_state.df['클래스'] == i]['cos'].values[0]), label_visibility='collapsed')
                    st.metric(label='_', value = 'SSIM', delta='-' + str(st.session_state.df[st.session_state.df['클래스'] == i]['ssim'].values[0]), label_visibility='collapsed')
                    st.markdown("<p style='text-align: center; font-size:10%'>. </p>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; font-size:10%'>. </p>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; font-size:10%'>. </p>", unsafe_allow_html=True)


        with last_co:
            st.markdown("<p style='text-align: center; color: red; font-size:120%'> Compared data </p>", unsafe_allow_html=True)
            for i in stqdm(st.session_state.df['클래스']):
                st.markdown("<p style='text-align: center; font-size:90%'> </p>", unsafe_allow_html=True)
                st.image(os.path.join(st.session_state.recon_data, str(i) ,os.path.basename(st.session_state.image)), width=225, caption= st.session_state['df'][st.session_state['df']['클래스'] == i]['클래스_이름'].values[0] + ': ' + str(i) + 'Class')


        # 폴더에 들어간 이미지와 레지스트 폴더 전체 삭제
        os.remove(os.path.join(os.getcwd(), 'dataset', os.path.basename(st.session_state.image)))
        shutil.rmtree(os.path.join(os.getcwd(), 'dataset', 'regist'))




    else:
        st.error("Please drag and drop file.")
        st.write(os.listdir(os.path.join(os.getcwd(), 'dataset')))
        if os.path.isdir(os.path.join(os.getcwd(), 'dataset', 'regist')):
            shutil.rmtree(os.path.join(os.getcwd(), 'dataset', 'regist'))

        if len(os.listdir(os.path.join(os.getcwd(), 'dataset'))) >= 2:
            os.remove(os.path.join(os.getcwd(), 'dataset', (os.listdir(os.path.join(os.getcwd(), 'dataset'))[0])))


