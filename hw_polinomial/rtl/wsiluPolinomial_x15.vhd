library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.float_pkg.all;

entity wsiluPolinomial_x15 is
	port (
		clk 	:	in 	std_logic;
		enable	:	in 	std_logic;
		reset	:	in 	std_logic;
		xIn1	: 	in 	float16;
        xIn2	: 	in 	float16;
        xIn3	: 	in 	float16;
        xIn4	: 	in 	float16;
        xIn5	: 	in 	float16;
        xIn6	: 	in 	float16;
        xIn7	: 	in 	float16;
        xIn8	: 	in 	float16;
        xIn9	: 	in 	float16;
        xIn10	: 	in 	float16;
        xIn11	: 	in 	float16;
        xIn12	: 	in 	float16;
        xIn13	: 	in 	float16;
        xIn14	: 	in 	float16;
        xIn15	: 	in 	float16;
		yOut1	: 	out float16;
        yOut2	: 	out float16;
        yOut3	: 	out float16;
        yOut4	: 	out float16;
        yOut5	: 	out float16;
        yOut6	: 	out float16;
        yOut7	: 	out float16;
        yOut8	: 	out float16;
        yOut9	: 	out float16;
        yOut10	: 	out float16;
        yOut11	: 	out float16;
        yOut12	: 	out float16;
        yOut13	: 	out float16;
        yOut14	: 	out float16;
        yOut15	: 	out float16
	);
end entity wsiluPolinomial_x15;

architecture wsiluPolinomial_x15_arch of wsiluPolinomial_x15 is

    component WSiluPolinomial is
		port (
			clk 	:	in 	std_logic;
			enable	:	in 	std_logic;
			reset	:	in 	std_logic;
			xIn		: 	in 	float16;
			yOut	: 	out float16
			);
	end component;

    signal	x1	: float16;
    signal	x2	: float16;
    signal	x3	: float16;
    signal	x4	: float16;
    signal	x5	: float16;
    signal	x6	: float16;
    signal	x7	: float16;
    signal	x8	: float16;
    signal	x9	: float16;
    signal	x10	: float16;
    signal	x11	: float16;
    signal	x12	: float16;
    signal	x13	: float16;
    signal	x14	: float16;
    signal	x15	: float16;

    signal	y1	: float16;
    signal	y2	: float16;
    signal	y3	: float16;
    signal	y4	: float16;
    signal	y5	: float16;
    signal	y6	: float16;
    signal	y7	: float16;
    signal	y8	: float16;
    signal	y9	: float16;
    signal	y10	: float16;
    signal	y11	: float16;
    signal	y12	: float16;
    signal	y13	: float16;
    signal	y14	: float16;
    signal	y15	: float16;
    
begin
    
    wsilu1: WSiluPolinomial port map(clk, enable, reset, x1, y1);
    wsilu2: WSiluPolinomial port map(clk, enable, reset, x2, y2);
    wsilu3: WSiluPolinomial port map(clk, enable, reset, x3, y3);
    wsilu4: WSiluPolinomial port map(clk, enable, reset, x4, y4);
    wsilu5: WSiluPolinomial port map(clk, enable, reset, x5, y5);
    wsilu6: WSiluPolinomial port map(clk, enable, reset, x6, y6);
    wsilu7: WSiluPolinomial port map(clk, enable, reset, x7, y7);
    wsilu8: WSiluPolinomial port map(clk, enable, reset, x8, y8);
    wsilu9: WSiluPolinomial port map(clk, enable, reset, x9, y9);
    wsilu10: WSiluPolinomial port map(clk, enable, reset, x10, y10);
    wsilu11: WSiluPolinomial port map(clk, enable, reset, x11, y11);
    wsilu12: WSiluPolinomial port map(clk, enable, reset, x12, y12);
    wsilu13: WSiluPolinomial port map(clk, enable, reset, x13, y13);
    wsilu14: WSiluPolinomial port map(clk, enable, reset, x14, y14);
    wsilu15: WSiluPolinomial port map(clk, enable, reset, x15, y15);

-- registrador de entrada
	process (reset, clk)
	begin
		if (reset = '1') then
			x1 <= (others => '0');
            x2 <= (others => '0');
            x3 <= (others => '0');
            x4 <= (others => '0');
            x5 <= (others => '0');
            x6 <= (others => '0');
            x7 <= (others => '0');
            x8 <= (others => '0');
            x9 <= (others => '0');
            x10 <= (others => '0');
            x11 <= (others => '0');
            x12 <= (others => '0');
            x13 <= (others => '0');
            x14 <= (others => '0');
            x15 <= (others => '0');
		elsif (clk'event and clk = '1') then
			x1 <= xIn1;
            x2 <= xIn2;
            x3 <= xIn3;
            x4 <= xIn4;
            x5 <= xIn5;
            x6 <= xIn6;
            x7 <= xIn7;
            x8 <= xIn8;
            x9 <= xIn9;
            x10 <= xIn10;
            x11 <= xIn11;
            x12 <= xIn12;
            x13 <= xIn13;
            x14 <= xIn14;
            x15 <= xIn15;
		end if;
	end process;

-- registrador de saída
	process (reset, clk)
	begin
		if (reset = '1') then
			yOut1 <= (others => '0');
            yOut2 <= (others => '0');
            yOut3 <= (others => '0');
            yOut4 <= (others => '0');
            yOut5 <= (others => '0');
            yOut6 <= (others => '0');
            yOut7 <= (others => '0');
            yOut8 <= (others => '0');
            yOut9 <= (others => '0');
            yOut10 <= (others => '0');
            yOut11 <= (others => '0');
            yOut12 <= (others => '0');
            yOut13 <= (others => '0');
            yOut14 <= (others => '0');
            yOut15 <= (others => '0');
		elsif (clk'event and clk = '1') then
			yOut1 <= y1;
            yOut2 <= y2;
            yOut3 <= y3;
            yOut4 <= y4;
            yOut5 <= y5;
            yOut6 <= y6;
            yOut7 <= y7;
            yOut8 <= y8;
            yOut9 <= y9;
            yOut10 <= y10;
            yOut11 <= y11;
            yOut12 <= y12;
            yOut13 <= y13;
            yOut14 <= y14;
            yOut15 <= y15;
		end if;
	end process;
    
end architecture wsiluPolinomial_x15_arch;