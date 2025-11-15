function [NLfinger,finger_stock] = Get_NLfinger_F(Load_Data_str,Save_Feature_str)

%% Parameters
load('Tx_OFDM_preamble.mat');
load(Load_Data_str);
finger_stock = cell(1,41);
Temp_1 = size(Data);
Segment_Number = Temp_1(1,1)-sum(abs(Data(:,1))==0);
Store_NLfinger = zeros(Segment_Number,21);
Get_count_point = zeros(1,Segment_Number);

%% 构造大D矩阵
sig=Tx_OFDM_preamble.';
D=zeros(320,2);
D(:,1)=sig;
D(:,2)=abs(sig).^2.*sig;
Segment_count = 1;
%% 提取指纹的循环
for Segment_Process = 1:Temp_1(1,1)
    
    preamble_sig=Data(Segment_Process,:).';
    
    if(abs(preamble_sig(1))>0)
        
        %------------------[L,R]范围设为[5,45]，遍历区间内每个值，计算对应的指纹值---------------%
        NLfinger=zeros(21,41);                      %存放对应的指纹值
        tag=1;
        for k=5:45
            len_F=2*(k+1);
            matrix_X=zeros(320,len_F);                           %以D0构造每个值对应的大D矩阵 用matrix_X表示
            for m=0:k
                matrix_X(m+1:end,(2*m+1):(2*m+2))=D(1:320-m,:);
            end
            finger=pinv(matrix_X)*preamble_sig;                %LS估计
            finger_stock{k-4} = finger;
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
            NLfinger(5,tag)=(sum(preamble_sig(33:256,1))+sum(value192_len-value32_len))/(sum(preamble_sig(33:224,1))+sum(value224_len-value32_len));
            NLfinger(6,tag)=(sum(preamble_sig(33:224,1))+sum(value224_len-value32_len))/(sum(preamble_sig(33:320,1))+sum(value192_len-value32_len));
            NLfinger(7,tag)=(sum(preamble_sig(33:160,1))+sum(value32_len-value32_len))/(sum(preamble_sig(33:256,1))+sum(value192_len-value32_len));
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
        
        value_var=zeros(1,36);
        choose_point=1;
        for kk=1:36
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

figure,
scatter(real(Store_NLfinger (:,1)),imag(Store_NLfinger (:,1)))
hold on
scatter(real(Store_NLfinger (:,2)),imag(Store_NLfinger (:,2)))
scatter(real(Store_NLfinger (:,3)),imag(Store_NLfinger (:,3)))
scatter(real(Store_NLfinger (:,4)),imag(Store_NLfinger (:,4)))
scatter(real(Store_NLfinger (:,5)),imag(Store_NLfinger (:,5)))
scatter(real(Store_NLfinger (:,6)),imag(Store_NLfinger (:,6)))
scatter(real(Store_NLfinger (:,7)),imag(Store_NLfinger (:,7)))
scatter(real(Store_NLfinger (:,8)),imag(Store_NLfinger (:,8)))
scatter(real(Store_NLfinger (:,9)),imag(Store_NLfinger (:,9)))
scatter(real(Store_NLfinger (:,10)),imag(Store_NLfinger (:,10)))
scatter(real(Store_NLfinger (:,11)),imag(Store_NLfinger (:,11)))
scatter(real(Store_NLfinger (:,12)),imag(Store_NLfinger (:,12)))
scatter(real(Store_NLfinger (:,13)),imag(Store_NLfinger (:,13)))
scatter(real(Store_NLfinger (:,14)),imag(Store_NLfinger (:,14)))
scatter(real(Store_NLfinger (:,15)),imag(Store_NLfinger (:,15)))
scatter(real(Store_NLfinger (:,16)),imag(Store_NLfinger (:,16)))
scatter(real(Store_NLfinger (:,17)),imag(Store_NLfinger (:,17)))
scatter(real(Store_NLfinger (:,18)),imag(Store_NLfinger (:,18)))
scatter(real(Store_NLfinger (:,19)),imag(Store_NLfinger (:,19)))
scatter(real(Store_NLfinger (:,20)),imag(Store_NLfinger (:,20)))
scatter(real(Store_NLfinger (:,21)),imag(Store_NLfinger (:,21)))
legend
title(Save_Feature_str)
size(Store_NLfinger)
save(Save_Feature_str,'Store_NLfinger','Get_count_point')




% SYF
% %--------------------计算对应的卷积残余-----------------------------------------------%
%      matrix_17_len=matrix_X(17:17+k-1,:);          %整周期短导码卷积残余对应的D
%      matrix_25_len=matrix_X(25:25+k-1,:);          %半周期短导码卷积残余对应的D
%      matrix_192_len=matrix_X(192:192+k-1,:);       %整周期长导码卷积残余对应的D
%       for n=0:k
%           matrix_17_len(n+1:end,2*n+1:2*n+2)=0;
%           matrix_25_len(n+1:end,2*n+1:2*n+2)=0;
%           matrix_192_len(n+1:end,2*n+1:2*n+2)=0;
%       end
%       value17_len=matrix_17_len* finger;          %整周期短导码卷积残余
%       value25_len=matrix_25_len* finger;          %半周期短导码卷积残余
%       value192_len=matrix_192_len* finger;        %整周期长导码卷积残余
%
%       NLfinger(16,tag)=((sum(preamble_sig(32:136,1))+sum(value25_len)))/((sum(preamble_sig(32:320,1))+sum(value17_len)));
%       NLfinger(17,tag)=((sum(preamble_sig(32:120,1))+sum(value25_len)))/((sum(preamble_sig(32:320,1))+sum(value25_len)));
%       NLfinger(18,tag)=((sum(preamble_sig(32:136,1))+sum(value25_len)))/((sum(preamble_sig(32:256,1))+sum(value17_len)));
%       NLfinger(19,tag)=((sum(preamble_sig(32:104,1))+sum(value25_len)))/((sum(preamble_sig(32:256,1))+sum(value17_len)));
%       NLfinger(20,tag)=((sum(preamble_sig(32:88,1))+sum(value25_len)))/((sum(preamble_sig(32:192,1))+sum(value17_len)));
%       NLfinger(21,tag)=((sum(preamble_sig(32:104,1))+sum(value25_len)))/((sum(preamble_sig(32:160,1))+sum(value17_len)));
%
%       NLfinger(10,tag)=((sum(preamble_sig(32:192,1))+sum(value25_len)))/((sum(preamble_sig(64:288,1))+sum(value25_len)));
%       NLfinger(11,tag)=((sum(preamble_sig(32:160,1))+sum(value25_len)))/((sum(preamble_sig(64:288,1))+sum(value25_len)));
%       NLfinger(12,tag)=((sum(preamble_sig(64:192,1))+sum(value25_len)))/((sum(preamble_sig(32:224,1))+sum(value25_len)));
%       NLfinger(13,tag)=((sum(preamble_sig(64:160,1))+sum(value25_len)))/((sum(preamble_sig(64:224,1))+sum(value25_len)));
%       NLfinger(14,tag)=((sum(preamble_sig(32:160,1))+sum(value25_len)))/((sum(preamble_sig(32:192,1))+sum(value25_len)));
%       NLfinger(15,tag)=((sum(preamble_sig(64:160,1))+sum(value25_len)))/((sum(preamble_sig(32:192,1))+sum(value25_len)));
% % % %
%    %-------------------------------------------余下六个指纹-----------------------------%
%
%       NLfinger(4,tag)=(((sum(preamble_sig(32:160,1))+sum(value17_len)))/((sum(preamble_sig(32:320,1))+sum(value25_len))));
%       NLfinger(5,tag)=(((sum(preamble_sig(32:256,1))+sum(value17_len)))/((sum(preamble_sig(32:320,1))+sum(value25_len))));
%       NLfinger(6,tag)=(((sum(preamble_sig(32:224,1))+sum(value17_len)))/((sum(preamble_sig(32:320,1))+sum(value25_len))));
%       NLfinger(7,tag)=(((sum(preamble_sig(32:160,1))+sum(value17_len)))/((sum(preamble_sig(32:256,1))+sum(value192_len))));
%       NLfinger(8,tag)=(((sum(preamble_sig(32:192,1))+sum(value192_len)))/((sum(preamble_sig(32:256,1))+sum(value192_len))));
%       NLfinger(9,tag)=(((sum(preamble_sig(32:256,1))+sum(value192_len)))/((sum(preamble_sig(32:320,1))+sum(value192_len))));
%  %%