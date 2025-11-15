function Get_NLfinger_raw_rebuildCSI(Load_str,Save_str)

%% Parameters
addpath('Functions');
load('OFDM_STF_LTF.mat');
load('Tx_OFDM_preamble.mat');
load(Load_str)
Raw_sig = Store_Data;
Temp_1 = size(Raw_sig);
Segment_Number = Temp_1(1,2);

%for Segment_Process = 1:200 %Segment_Number
for Segment_Process = 1:200
    CSI_mat = Raw_sig{1, Segment_Process}.CSI;
    for ii = 1 %:4
        CSI_vect = CSI_mat(:,ii);
        LTS1 = LTF_Pattern(7:32).'.*CSI_vect(1:26);
        LTS2 = LTF_Pattern(34:59).'.*CSI_vect(27:52);
        sig_rebuild = ifft([0;LTS2;zeros(11,1);LTS1]);
        sig_d = OFDM_LTF.';
        %------------------[L,R]范围设为[5,45]，遍历区间内每个值，计算对应的指纹值---------------%
        NLfinger=zeros(21,20);                      %存放对应的指纹值
        tag=1;
        for k=10:32
            len_F=2*(k+1);
            matrix_X=zeros(64,len_F);                           %以D0构造每个值对应的大D矩阵 用matrix_X表示
            matrix_X2=zeros(128,len_F); 
            for m=0:k
                matrix_X(:,(2*m+1):(2*m+2))=[sig_d(33-m:96-m) abs(sig_d(33-m:96-m)).^2.*sig_d(33-m:96-m)];
                matrix_X2(:,(2*m+1):(2*m+2))=[sig_d(33-m:160-m) abs(sig_d(33-m:160-m)).^2.*sig_d(33-m:160-m)];
            end
            finger=pinv(matrix_X)*sig_rebuild;                %LS估计

            %-------------------计算前三个指纹值--------------------------------------------------%
            NLfinger(1,tag)= finger(1)/ finger(2);
            NLfinger(2,tag)= finger(end-1)/ finger(end);
            NLfinger(3,tag)=sum(finger(1:2:(len_F-1),:))/sum(finger(2:2:len_F,:) );

            matrix_32_tail=matrix_X(33:33+k-1,:);
            matrix_64_tail=matrix_X2(65:65+k-1,:);

            for n=0:k-1
                matrix_32_tail(n+1:end,2*n+1:2*n+2)=0;
                matrix_64_tail(n+1:end,2*n+1:2*n+2)=0;
            
            end

            value32_len = matrix_32_tail* finger; 
            value64_len = matrix_64_tail* finger; 
          %  NLfinger(4,tag)=(sum(sig_rebuild(1:32))+sum(value32_len-value64_len))/(sum(sig_rebuild));
            % NLfinger(5,tag)=(sum(preamble_sig(33:160,1))+sum(value32_len-value32_len))/(sum(preamble_sig(33:256,1))+sum(value192_len-value32_len));
            % NLfinger(6,tag)=(sum(preamble_sig(33:256,1))+sum(value192_len-value32_len))/(sum(preamble_sig(33:224,1))+sum(value224_len-value32_len));
            % NLfinger(7,tag)=(sum(preamble_sig(33:224,1))+sum(value224_len-value32_len))/(sum(preamble_sig(33:320,1))+sum(value192_len-value32_len));
            %
            % NLfinger(8,tag)=(sum(preamble_sig(33:192,1))+sum(value192_len-value32_len))/(sum(preamble_sig(33:256,1))+sum(value192_len-value32_len));
            % NLfinger(9,tag)=(sum(preamble_sig(33:256,1))+sum(value192_len-value32_len))/(sum(preamble_sig(33:320,1))+sum(value192_len-value32_len));

            %%
            % matrix_32_tail=matrix_X(33:33+k-1,:);          %整周期短导码卷积残余对应的D
            % matrix_40_tail=matrix_X(41:41+k-1,:);          %半周期短导码卷积残余对应的D
            % matrix_192_tail=matrix_X(193:193+k-1,:);       %半周期长导码卷积残余对应的D
            % matrix_224_tail=matrix_X(225:225+k-1,:);       %半周期长导码卷积残余对应的D
            % for n=0:k-1
            %     matrix_32_tail(n+1:end,2*n+1:2*n+2)=0;
            %     matrix_40_tail(n+1:end,2*n+1:2*n+2)=0;
            %     matrix_192_tail(n+1:end,2*n+1:2*n+2)=0;
            %     matrix_224_tail(n+1:end,2*n+1:2*n+2)=0;
            % end
            % value32_len= matrix_32_tail* finger;          %整周期短导码卷积残余
            % value40_len= matrix_40_tail* finger;          %半周期短导码卷积残余
            % value192_len=matrix_192_tail* finger;         %整周期长导码卷积残余
            % value224_len=matrix_224_tail* finger;         %整周期长导码卷积残余
            %
            % %--------------------计算对应的卷积残余头部（从32位置处开始截取）----------------------------------------------------------------------------------%
            % NLfinger(16,tag)=(sum(preamble_sig(33:136,1))+sum(value40_len-value32_len))/(sum(preamble_sig(33:320,1))+sum(value192_len-value32_len));
            % NLfinger(17,tag)=(sum(preamble_sig(33:120,1))+sum(value40_len-value32_len))/(sum(preamble_sig(33:320,1))+sum(value192_len-value32_len));
            % NLfinger(18,tag)=(sum(preamble_sig(33:136,1))+sum(value40_len-value32_len))/(sum(preamble_sig(33:256,1))+sum(value192_len-value32_len));
            % NLfinger(19,tag)=(sum(preamble_sig(33:104,1))+sum(value40_len-value32_len))/(sum(preamble_sig(33:256,1))+sum(value192_len-value32_len));
            % NLfinger(20,tag)=(sum(preamble_sig(33:88,1))+sum(value40_len-value32_len))/(sum(preamble_sig(33:192,1))+sum(value192_len-value32_len));
            % NLfinger(21,tag)=(sum(preamble_sig(33:104,1))+sum(value40_len-value32_len))/(sum(preamble_sig(33:160,1))+sum(value32_len-value32_len));
            %
            % NLfinger(10,tag)=(sum(preamble_sig(33:192,1))+sum(value192_len-value32_len))/(sum(preamble_sig(65:288,1))+sum(value224_len-value32_len));
            % NLfinger(11,tag)=(sum(preamble_sig(33:160,1))+sum(value32_len-value32_len))/(sum(preamble_sig(65:288,1))+sum(value224_len-value32_len));
            % NLfinger(12,tag)=(sum(preamble_sig(65:192,1))+sum(value192_len-value32_len))/(sum(preamble_sig(33:224,1))+sum(value224_len-value32_len));
            % NLfinger(13,tag)=(sum(preamble_sig(65:160,1))+sum(value32_len-value32_len))/(sum(preamble_sig(65:224,1))+sum(value224_len-value32_len));
            % NLfinger(14,tag)=(sum(preamble_sig(33:160,1))+sum(value32_len-value32_len))/(sum(preamble_sig(33:192,1))+sum(value192_len-value32_len));
            % NLfinger(15,tag)=(sum(preamble_sig(65:160,1))+sum(value32_len-value32_len))/(sum(preamble_sig(33:192,1))+sum(value192_len-value32_len));
            % %    %-------------------------------------------余下六个指纹-----------------------------%
            tag=tag+1;
        end

        value_var=zeros(1,19);
        choose_point=1;
        for kk=1:19
            value_var(1,kk)=var(NLfinger(3,kk:kk+4));  %计算连续4个值对应的方差，找到满足阈值的点，作为估计值
            %             if(kk>1) && (7*value_var(1,kk)<value_var(1,kk-1))
            %                 choose_point=kk;
            %                 %  break;
            %             end
        end

        %figure,plot(value_var(1:15))
        % for nn=1:(length(value_var)-4)
        %     if (value_var(1,nn)<0.00015) && (sum(value_var(1,nn:nn+3)>0.00015)<1)
        %         choose_point=nn;
        %         break
        %     end
        % end

        % if choose_point<10
        %     %[M,I] = min(value_var);
        %     choose_point = 20;
        % end

        %         if Segment_count<20
        %             figure,plot(value_var)
        %         end
        Store_NLfinger(Segment_Process,ii,:)=NLfinger(:,10); % D39p1-20 % D39p2-20 D41p2-20

    end
end

%save(Save_str,'Store_NLfinger','Get_count_point')
figure,
scatter(real(Store_NLfinger(:,ii,1)),imag(Store_NLfinger(:,ii,1)))
hold on
for pt_Index = 2:3
    scatter(real(Store_NLfinger(:,ii,pt_Index)),imag(Store_NLfinger(:,ii,pt_Index)))
end
hold off