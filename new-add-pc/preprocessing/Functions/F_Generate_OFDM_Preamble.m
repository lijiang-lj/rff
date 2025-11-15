function [OFDM_STF, OFDM_LTF] = F_Generate_OFDM_Preamble()
    global STF_Pattern LTF_Pattern
    OFDM_FFT_Length = 64;
    OFDM_CP_Length = 16;
    OFDM_Length = OFDM_FFT_Length + OFDM_CP_Length;
    OFDM_STF_Symbol = 2;
    OFDM_LTF_Symbol = 2;

    OFDM_Temp = zeros(1,OFDM_Length);
    OFDM_STF_Length = OFDM_Length * OFDM_STF_Symbol;
    OFDM_LTF_Length = OFDM_Length * OFDM_LTF_Symbol;
    OFDM_STF = zeros(1,OFDM_STF_Length);
    OFDM_LTF = zeros(1,OFDM_LTF_Length);

    STF_Pattern = [0 0 0 0 0 0 0 0 ...
                  1+1i 0 0 0 -1-1i 0 0 0 ...
                  1+1i 0 0 0 -1-1i 0 0 0 ...
                  -1-1i 0 0 0 1+1i 0 0 0 ...
                  0 0 0 0 -1-1i 0 0 0 ...
                  -1-1i 0 0 0 1+1i 0 0 0 ...
                  1+1i 0 0 0 1+1i 0 0 0 ...
                  1+1i 0 0 0 0 0 0 0];
    STF_Factor = sqrt(13/6);

    LTF_Pattern = [0 0 0 0 0 0 1 1 ...
                  -1 -1 1 1 -1 1 -1 1 ...
                  1 1 1 1 1 -1 -1 1 ...
                  1 -1 1 -1 1 1 1 1 ...
                  0 1 -1 -1 1 1 -1 1 ...
                  -1 1 -1 -1 -1 -1 -1 1 ...
                  1 -1 -1 1 -1 1 -1 1 ...
                  1 1 1 0 0 0 0 0];

    OFDM_Symbol_STF = STF_Factor * STF_Pattern;
    OFDM_Symbol_STF = circshift(OFDM_Symbol_STF,[0,32]);
    OFDM_Symbol_STF = ifft(OFDM_Symbol_STF);
    OFDM_Temp(1,1 + OFDM_CP_Length:OFDM_Length) = OFDM_Symbol_STF(1,1:OFDM_FFT_Length);
    OFDM_Temp(1,1:OFDM_CP_Length) = OFDM_Symbol_STF(1,OFDM_FFT_Length - OFDM_CP_Length + 1:OFDM_FFT_Length);
    Temp_1 = 0;
    for n = 1:OFDM_STF_Symbol
        OFDM_STF(1,Temp_1+1:Temp_1+OFDM_Length) = OFDM_Temp(1,1:OFDM_Length);
        Temp_1 = Temp_1 + OFDM_Length;
    end

    OFDM_Symbol_LTF = LTF_Pattern;
    OFDM_Symbol_LTF = circshift(OFDM_Symbol_LTF,[0,32]);
    OFDM_Symbol_LTF = ifft(OFDM_Symbol_LTF);
    OFDM_LTF(1,1:2*OFDM_CP_Length) = OFDM_Symbol_LTF(1,OFDM_FFT_Length - 2 * OFDM_CP_Length + 1:OFDM_FFT_Length);
    OFDM_LTF(1,2*OFDM_CP_Length+1:2*OFDM_CP_Length+OFDM_FFT_Length) = OFDM_Symbol_LTF(1,1:OFDM_FFT_Length);
    OFDM_LTF(1,2*OFDM_CP_Length+OFDM_FFT_Length+1:2*OFDM_CP_Length+2*OFDM_FFT_Length) = OFDM_Symbol_LTF(1,1:OFDM_FFT_Length);
    
