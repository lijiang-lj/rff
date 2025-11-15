function [WiFi_Data_Compensated,Compensation_Phase_Store,Estimated_EVM] = F_Get_WIFI_Data_Compensation(WiFi_Data)
    WiFi_Data_Length = length(WiFi_Data);
    
    if(WiFi_Data_Length>1)
        WiFi_Data_Compensated = WiFi_Data;
        OFDM_Symbol_Number = WiFi_Data_Length / 53;
        Compensation_Phase_Store = zeros(1,OFDM_Symbol_Number);
        for OFDM_Proess_Index = 1:OFDM_Symbol_Number
            Temp_1 = WiFi_Data_Compensated((OFDM_Proess_Index-1)*53+6);
            Temp_2 = WiFi_Data_Compensated((OFDM_Proess_Index-1)*53+20);
            Temp_3 = WiFi_Data_Compensated((OFDM_Proess_Index-1)*53+34);
            Temp_4 = WiFi_Data_Compensated((OFDM_Proess_Index-1)*53+48);
            Temp_1 = Temp_1 / abs(Temp_1);
            Temp_2 = Temp_2 / abs(Temp_2);
            Temp_3 = Temp_3 / abs(Temp_3);
            Temp_4 = Temp_4 / abs(Temp_4);
            Temp_5 = (Temp_1 + Temp_2 + Temp_3 - Temp_4)/4;
            Temp_5 = angle(Temp_5);
            for n = 1:53
                WiFi_Data_Compensated((OFDM_Proess_Index-1)*53+n) = WiFi_Data_Compensated((OFDM_Proess_Index-1)*53+n) * exp(-1i*Temp_5);
            end
            Compensation_Phase_Store(1,OFDM_Proess_Index) = Temp_5;
        end

        Temp_1 = 0;
        Temp_2 = 0;
        Temp_3 = 0;
        Temp_4 = 0;
        for n = 1:26
            if(real(WiFi_Data_Compensated(n))>0)
                Temp_1 = Temp_1 + WiFi_Data_Compensated(n);
                Temp_2 = Temp_2 + 1;
            else
                Temp_3 = Temp_3 + WiFi_Data_Compensated(n);
                Temp_4 = Temp_4 + 1;
            end
        end
        for n = 28:53
            if(real(WiFi_Data_Compensated(n))>0)
                Temp_1 = Temp_1 + WiFi_Data_Compensated(n);
                Temp_2 = Temp_2 + 1;
            else
                Temp_3 = Temp_3 + WiFi_Data_Compensated(n);
                Temp_4 = Temp_4 + 1;
            end
        end

        Temp_1 = Temp_1 / Temp_2;
        Temp_3 = Temp_3 / Temp_4;

        Temp_2 = 0;
        for n = 1:26
            if(real(WiFi_Data_Compensated(n))>0)
                Temp_2 = Temp_2 + (abs(WiFi_Data_Compensated(n) - Temp_1))^2;
            else
                Temp_2 = Temp_2 + (abs(WiFi_Data_Compensated(n) - Temp_3))^2;
            end
        end
        for n = 28:53
            if(real(WiFi_Data_Compensated(n))>0)
                Temp_2 = Temp_2 + (abs(WiFi_Data_Compensated(n) - Temp_1))^2;
            else
                Temp_2 = Temp_2 + (abs(WiFi_Data_Compensated(n) - Temp_3))^2;
            end
        end
        Temp_2 = Temp_2 / 52;

        Estimated_EVM = 10*log10(((abs(Temp_1))^2 + (abs(Temp_3))^2)/2/Temp_2);
    else
        WiFi_Data_Compensated = WiFi_Data;
        Compensation_Phase_Store = 0;
        Estimated_EVM = 0;
    end