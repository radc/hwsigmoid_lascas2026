library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.float_pkg.all;
use work.fixed_pkg.all;
use std.textio.all;

entity testbench_x30 is
end testbench_x30;

architecture testbench_arc of testbench_x30 is

	signal clk 		:	std_logic;
	signal enable	:	std_logic;
	signal reset	:	std_logic;

	signal xIn1		: 	float16;
	signal xIn2		: 	float16;
	signal xIn3		: 	float16;
	signal xIn4		: 	float16;
	signal xIn5		: 	float16;
	signal xIn6		: 	float16;
	signal xIn7		: 	float16;
	signal xIn8		: 	float16;
	signal xIn9		: 	float16;
	signal xIn10	: 	float16;
	signal xIn11	: 	float16;
	signal xIn12	: 	float16;
	signal xIn13	: 	float16;
	signal xIn14	: 	float16;
	signal xIn15	: 	float16;
	signal xIn16	: 	float16;
	signal xIn17	: 	float16;
	signal xIn18	: 	float16;
	signal xIn19	: 	float16;
	signal xIn20	: 	float16;
	signal xIn21	: 	float16;
	signal xIn22	: 	float16;
	signal xIn23	: 	float16;
	signal xIn24	: 	float16;
	signal xIn25	: 	float16;
	signal xIn26	: 	float16;
	signal xIn27	: 	float16;
	signal xIn28	: 	float16;
	signal xIn29	: 	float16;
	signal xIn30	: 	float16;
	signal yOut1	: 	float16;
	signal yOut2	: 	float16;
	signal yOut3	: 	float16;
	signal yOut4	: 	float16;
	signal yOut5	: 	float16;
	signal yOut6	: 	float16;
	signal yOut7	: 	float16;
	signal yOut8	: 	float16;
	signal yOut9	: 	float16;
	signal yOut10	: 	float16;
	signal yOut11	: 	float16;
	signal yOut12	: 	float16;
	signal yOut13	: 	float16;
	signal yOut14	: 	float16;
	signal yOut15	: 	float16;
	signal yOut16	: 	float16;
	signal yOut17	: 	float16;
	signal yOut18	: 	float16;
	signal yOut19	: 	float16;
	signal yOut20	: 	float16;
	signal yOut21	: 	float16;
	signal yOut22	: 	float16;
	signal yOut23	: 	float16;
	signal yOut24	: 	float16;
	signal yOut25	: 	float16;
	signal yOut26	: 	float16;
	signal yOut27	: 	float16;
	signal yOut28	: 	float16;
	signal yOut29	: 	float16;
	signal yOut30	: 	float16;

	signal nTestes: integer := 0;
	signal acabou: integer := 0;

	component WSiluPolinomial_x30 is
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
			xIn16	: 	in 	float16;
			xIn17	: 	in 	float16;
			xIn18	: 	in 	float16;
			xIn19	: 	in 	float16;
			xIn20	: 	in 	float16;
			xIn21	: 	in 	float16;
			xIn22	: 	in 	float16;
			xIn23	: 	in 	float16;
			xIn24	: 	in 	float16;
			xIn25	: 	in 	float16;
			xIn26	: 	in 	float16;
			xIn27	: 	in 	float16;
			xIn28	: 	in 	float16;
			xIn29	: 	in 	float16;
			xIn30	: 	in 	float16;
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
			yOut15	: 	out float16;
			yOut16	: 	out float16;
			yOut17	: 	out float16;
			yOut18	: 	out float16;
			yOut19	: 	out float16;
			yOut20	: 	out float16;
			yOut21	: 	out float16;
			yOut22	: 	out float16;
			yOut23	: 	out float16;
			yOut24	: 	out float16;
			yOut25	: 	out float16;
			yOut26	: 	out float16;
			yOut27	: 	out float16;
			yOut28	: 	out float16;
			yOut29	: 	out float16;
			yOut30	: 	out float16
		);
	end component;
--------------------------------------------------------------
	function string_to_float (s: string) return float16 is
		variable slv : std_logic_vector(15 downto 0);
		variable fp16 : float16;
	begin
		for i in 1 to 16 loop
			if(s(i) = '1') then
				slv(16-i) := '1';
			elsif(s(i) = '0') then
				slv(16-i) := '0';
			end if;
		end loop;
		for i in 0 to 15 loop
			fp16(5-i) := slv(15-i);
		end loop;
		return fp16; 
	end function string_to_float;
-------------------------------------------------------------
	function float_to_string(inp: float16) return string is
		variable temp: string(16 downto 1);
		variable slv: std_logic_vector(15 downto 0);
	begin
		for i in 0 to 15 loop
			slv(15-i) := inp(5-i);
		end loop;
		for i in 0 to 15 loop
			if (slv(i) = '1') then
				temp(i+1) := '1';
			elsif (slv(i) = '0') then
				temp(i+1) := '0';
			end if;
		end loop;
		return temp;
	end function float_to_string;

begin

	DUT: WSiluPolinomial_x30 port map(clk,enable,reset,xIn1,xIn2,xIn3,xIn4,xIn5,xIn6,xIn7,xIn8,xIn9,xIn10,xIn11,xIn12,xIn13,xIn14,xIn15,xIn16,xIn17,xIn18,xIn19,xIn20,xIn21,xIn22,xIn23,xIn24,xIn25,xIn26,xIn27,xIn28,xIn29,xIn30,yOut1,yOut2,yOut3,yOut4,yOut5,yOut6,yOut7,yOut8,yOut9,yOut10,yOut11,yOut12,yOut13,yOut14,yOut15,yOut16,yOut17,yOut18,yOut19,yOut20,yOut21,yOut22,yOut23,yOut24,yOut25,yOut26,yOut27,yOut28,yOut29,yOut30);

	clockProcess: process -- process para controlar o clock
	begin
		clk <= '0';
		wait for 5 ns;
		clk <= '1';
		wait for 5 ns;
	end process clockProcess;

	resetProcess: process -- process para controlar o reset
	begin
		reset <= '0';
		wait;
	end process resetProcess;

	enableProcess: process -- process para controlar o enable
	begin
		enable <= '0';
		wait for 7 ns;
		enable <= '1';
		wait;
	end process enableProcess;

	EntradasProcess: process (clk)

		file arquivo: text open read_mode is "entradasReais/entradasBin_x30.txt";
		variable linha: line;
		variable amostra: string(1 to 16);
		variable char: character;

		begin

			if (rising_edge(clk)) then
				if not(endfile(arquivo)) then	-- enquanto exixtirem linhas no arquivo atual
					readline(arquivo, linha);	-- lê uma linha e armazena em uma string/line (linha)

					-- lê os valores da linhas e passa para as entradas do circuito

					read(linha, amostra);
						xIn1 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn2 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn3 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn4 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn5 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn6 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn7 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn8 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn9 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn10 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn11 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn12 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn13 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn14 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn15 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn16 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn17 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn18 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn19 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn20 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn21 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn22 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn23 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn24 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn25 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn26 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn27 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn28 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn29 <= string_to_float(amostra); 
					read(linha, char);

					read(linha, amostra);
						xIn30 <= string_to_float(amostra); 
					read(linha, char);

					nTestes <= nTestes + 1;
				elsif (endfile(arquivo)) then
				--	assert false report "o arquivo acabou" severity failure;
					acabou <= acabou + 1;
				end if;
			end if;

	end process EntradasProcess;

	SaidaProcess: process(clk)

		begin

			if(acabou = 3) then 
				assert false report "o arquivo acabou" severity failure;
			end if;

	end process SaidaProcess;

end testbench_arc;