function Get_NLfinger_raw_F(Load_str,Save_str)

%% Parameters
addpath('FDE_functions');
load('OFDM_STF_LTF.mat');
load('Tx_OFDM_preamble.mat');
load(Load_str)
Raw_Date = Data;
Temp_1 = size(Raw_Date);
Segment_Number = Temp_1(1,1)-sum(abs(Raw_Date(:,1))==0);
Segment_Length = Temp_1(1,2);
Raw_Data = zeros(1,Segment_Length);
Get_Data_Length = 480;

Store_NLfinger = [];
Get_count_point = [];

%% 构造大D矩阵
sig=Tx_OFDM_preamble.';
D=zeros(320,2);
D(:,1)=sig;
D(:,2)=abs(sig).^2.*sig;
Segment_count = 1;

for Segment_Process = 1:Temp_1(1,1)
    
    if abs(Raw_Date(Segment_Process,1))>0
        %% Pre Processing
        Raw_Data(1,1:Segment_Length) = Raw_Date(Segment_Process,1:Segment_Length);
        Syn_Index_Coarse = 1;
        Corr_Find_Length = 100;
        [Syn_Index_Corr_Self] = F_Synchronization_Self_Corr(Raw_Data, Syn_Index_Coarse, 32, 64, 2, Corr_Find_Length);
        
        if(Syn_Index_Corr_Self>0 && Syn_Index_Corr_Self < 100)
            
            Get_Syn_Index = Syn_Index_Coarse + Syn_Index_Corr_Self - 4;
            
            %%% Get Data from File
            Get_OFDM_Data = zeros(1,Get_Data_Length);
            Get_OFDM_Data(1,1:Get_Data_Length) = Raw_Data(1,Get_Syn_Index+1:Get_Syn_Index+Get_Data_Length);
            
            %%% Estimate Frequency Offset
            [Est_Freq_Offset_Coarse] = F_Frequency_Offset_Estimation(Get_OFDM_Data, 0, 144, 16);
            [Get_OFDM_Data] = F_Frequency_Offset_Compensationn(Get_OFDM_Data, -Est_Freq_Offset_Coarse);
            
            preamble_sig = Get_OFDM_Data(1,1:320).';
            
            if(abs(preamble_sig(1))>0)
                
                %------------------[L,R]范围设为[5,45]，遍历区间内每个值，计算对应的指纹值---------------%
                NLfinger=zeros(21,36);                      %存放对应的指纹值
                tag=1;
                for k=5:40
                    len_F=2*(k+1);
                    matrix_X=zeros(320,len_F);                           %以D0构造每个值对应的大D矩阵 用matrix_X表示
                    for m=0:k
                        matrix_X(m+1:end,(2*m+1):(2*m+2))=D(1:320-m,:);
                    end
                    finger=pinv(matrix_X)*preamble_sig;                %LS估计
                    
                    %-------------------计算前三个指纹值--------------------------------------------------%
                    NLfinger(1,tag)= finger(1)/ finger(2);
                    NLfinger(2,tag)= finger(end-1)/ finger(end);
                    NLfinger(3,tag)=sum(finger(1:2:(len_F-1),:))/sum(finger(2:2:len_F,:) );
                    %%
                    matrix_32_tail=matrix_X(33:33+k-1,:);          %整周期短导码卷积残余对应的D
                    matrix_40_tail=matrix_X(41:41+k-1,:);          %半周期短导码卷积残余对应的D
                    matrix_192_tail=matrix_X(193:193+k-1,:);       %半周期长导码卷积残余对应的D
                    matrix_224_tail=matrix_X(225:225+k-1,:);       %半周期长导码卷积残余对应的D
                    for n=0:k-1
                        matrix_32_tail(n+1:end,2*n+1:2*n+2)=0;
                        matrix_40_tail(n+1:end,2*n+1:2*n+2)=0;
                        matrix_192_tail(n+1:end,2*n+1:2*n+2)=0;
                        matrix_224_tail(n+1:end,2*n+1:2*n+2)=0;
                    end
                    value32_len= matrix_32_tail* finger;          %整周期短导码卷积残余
                    value40_len= matrix_40_tail* finger;          %半周期短导码卷积残余
                    value192_len=matrix_192_tail* finger;         %整周期长导码卷积残余
                    value224_len=matrix_224_tail* finger;         %整周期长导码卷积残余
                    
                    %--------------------计算对应的卷积残余头部（从32位置处开始截取）----------------------------------------------------------------------------------%
                    NLfinger(16,tag)=(sum(preamble_sig(33:136,1))+sum(value40_len-value32_len))/(sum(preamble_sig(33:320,1))+sum(value192_len-value32_len));
                    NLfinger(17,tag)=(sum(preamble_sig(33:120,1))+sum(value40_len-value32_len))/(sum(preamble_sig(33:320,1))+sum(value192_len-value32_len));
                    NLfinger(18,tag)=(sum(preamble_sig(33:136,1))+sum(value40_len-value32_len))/(sum(preamble_sig(33:256,1))+sum(value192_len-value32_len));
                    NLfinger(19,tag)=(sum(preamble_sig(33:104,1))+sum(value40_len-value32_len))/(sum(preamble_sig(33:256,1))+sum(value192_len-value32_len));
                    NLfinger(20,tag)=(sum(preamble_sig(33:88,1))+sum(value40_len-value32_len))/(sum(preamble_sig(33:192,1))+sum(value192_len-value32_len));
                    NLfinger(21,tag)=(sum(preamble_sig(33:104,1))+sum(value40_len-value32_len))/(sum(preamble_sig(33:160,1))+sum(value32_len-value32_len));
                    
                    NLfinger(10,tag)=(sum(preamble_sig(33:192,1))+sum(value192_len-value32_len))/(sum(preamble_sig(65:288,1))+sum(value224_len-value32_len));
                    NLfinger(11,tag)=(sum(preamble_sig(33:160,1))+sum(value32_len-value32_len))/(sum(preamble_sig(65:288,1))+sum(value224_len-value32_len));
                    NLfinger(12,tag)=(sum(preamble_sig(65:192,1))+sum(value192_len-value32_len))/(sum(preamble_sig(33:224,1))+sum(value224_len-value32_len));
                    NLfinger(13,tag)=(sum(preamble_sig(65:160,1))+sum(value32_len-value32_len))/(sum(preamble_sig(65:224,1))+sum(value224_len-value32_len));
                    NLfinger(14,tag)=(sum(preamble_sig(33:160,1))+sum(value32_len-value32_len))/(sum(preamble_sig(33:192,1))+sum(value192_len-value32_len));
                    NLfinger(15,tag)=(sum(preamble_sig(65:160,1))+sum(value32_len-value32_len))/(sum(preamble_sig(33:192,1))+sum(value192_len-value32_len));
                    %    %-------------------------------------------余下六个指纹-----------------------------%
                    NLfinger(4,tag)=(sum(preamble_sig(33:160,1))+sum(value32_len-value32_len))/(sum(preamble_sig(33:320,1))+sum(value192_len-value32_len));
                    NLfinger(7,tag)=(sum(preamble_sig(33:160,1))+sum(value32_len-value32_len))/(sum(preamble_sig(33:256,1))+sum(value192_len-value32_len));
                    NLfinger(5,tag)=(sum(preamble_sig(33:256,1))+sum(value192_len-value32_len))/(sum(preamble_sig(33:224,1))+sum(value224_len-value32_len));
                    NLfinger(6,tag)=(sum(preamble_sig(33:224,1))+sum(value224_len-value32_len))/(sum(preamble_sig(33:320,1))+sum(value192_len-value32_len));
                   
                    NLfinger(8,tag)=(sum(preamble_sig(33:192,1))+sum(value192_len-value32_len))/(sum(preamble_sig(33:256,1))+sum(value192_len-value32_len));
                    NLfinger(9,tag)=(sum(preamble_sig(33:256,1))+sum(value192_len-value32_len))/(sum(preamble_sig(33:320,1))+sum(value192_len-value32_len));
                    
                    %       finger_correlation(4,tag)=(((sum(preamble_sig(1:160,1))+sum(value17_len)))/((sum(preamble_sig(1:152,1))+sum(value25_len))));
                    %       finger_correlation(5,tag)=(((sum(preamble_sig(1:80,1))+sum(value17_len)))/((sum(preamble_sig(1:136,1))+sum(value25_len))));
                    %       finger_correlation(6,tag)=(((sum(preamble_sig(1:144,1))+sum(value17_len)))/((sum(preamble_sig(1:152,1))+sum(value25_len))));
                    %       finger_correlation(7,tag)=(((sum(preamble_sig(1:160,1))+sum(value17_len)))/((sum(preamble_sig(1:256,1))+sum(value192_len))));
                    %       finger_correlation(8,tag)=(((sum(preamble_sig(1:192,1))+sum(value192_len)))/((sum(preamble_sig(1:256,1))+sum(value192_len))));
                    %       finger_correlation(9,tag)=(((sum(preamble_sig(1:256,1))+sum(value192_len)))/((sum(preamble_sig(1:320,1))+sum(value192_len))));
                    tag=tag+1;
                end
                
                value_var=zeros(1,32);
                choose_point=1;
                for kk=1:32
                    value_var(1,kk)=var(NLfinger(3,kk:kk+4));  %计算连续4个值对应的方差，找到满足阈值的点，作为估计值
                    %             if(kk>1) && (7*value_var(1,kk)<value_var(1,kk-1))
                    %                 choose_point=kk;
                    %                 %  break;
                    %             end
                end
                
                for nn=1:(length(value_var)-4)
                    if (value_var(1,nn)<0.00015) && (sum(value_var(1,nn:nn+3)>0.00015)<1)
                        choose_point=nn;
                        break
                    end
                end
                
                if choose_point<10
                    %[M,I] = min(value_var);
                    choose_point = 20;
                end
                
                %         if Segment_count<20
                %             figure,plot(value_var)
                %         end
                Store_NLfinger(Segment_count,:)=NLfinger(:,20).'; % D39p1-20 % D39p2-20 D41p2-20
                Get_count_point(1,Segment_count)=choose_point;
                Segment_count = Segment_count+1;
                
            end
        end
    end
end
save(Save_str,'Store_NLfinger','Get_count_point')
