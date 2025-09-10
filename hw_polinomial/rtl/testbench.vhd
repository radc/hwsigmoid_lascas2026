library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.float_pkg.all;
use work.fixed_pkg.all;
use std.textio.all;

entity testbench is
end testbench;

architecture testbench_arc of testbench is

	signal clk 		:	 std_logic;
	signal enable	:	 std_logic;
	signal reset	:	 std_logic;
	signal x		: 	 float16;
	signal y		: 	 float16;

	signal entrada: float16;
	signal saidaEsperada: float16;

	component WSiluPolinomial is
		port (
			clk 	:	in 	std_logic;
			enable	:	in 	std_logic;
			reset	:	in 	std_logic;
			xIn		: 	in 	float16;
			yOut	: 	out float16
			);
	end component;

	function string_to_float16 (s: string) return float16 is
		-- Variável para montar o vetor de bits intermediário
		variable result_slv : std_logic_vector(15 downto 0);
	begin

		-- 2. Loop para converter cada caractere da string em um bit
		for i in 2 to 16 loop
			-- O 'range da string é tipicamente (1 to 16).
			-- Mapeia string(1..16) para slv(15..0)
			-- O índice do vetor será: (s'length - i) + s'low - 1
			-- Para uma string(1 to 16), s'length=16, s'low=1, i=1..16 -> (16-i)+1-1 = 16-i
			-- Para uma string(0 to 15), s'length=16, s'low=0, i=0..15 -> (16-i)+0-1 = 15-i
			-- Vamos usar a forma mais simples (16 - i) assumindo range (1 to 16) por padrão.
			
			if s(i) = '1' then
				result_slv(16 - i) := '1'; -- Corrigido: operador '=' e aspas simples
			elsif s(i) = '0' then          -- Corrigido: 'elsif', '=', e aspas simples
				result_slv(16 - i) := '0';
			else
				-- Adiciona um erro para caracteres inválidos
				report "ERRO: Caractere inválido na string binária: " & s(i) severity failure;
			end if;
		end loop;
		
		-- 3. Faz o type cast (conversão de tipo) do std_logic_vector para float16 e retorna
		return to_float(unsigned(result_slv), 4, 10);
		
	end function string_to_float16;

begin

	DUT: WSiluPolinomial port map(clk, enable, reset, x, y);

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

		-- file saidaEsperada: text open write_mode is "saidaEsperada.txt";
		-- variable outEsperada: std_logic_vector(15 downto 0);
		-- variable str_outEsperada: string(18 downto 1);
		-- variable outlineEsperada: line;	

		-- file saidaGerada: text open write_mode is "saidaGerada.txt";
		-- variable outGerada: std_logic_vector(15 downto 0);
		-- variable str_outGerada: string(18 downto 1);
		-- variable outlineGerada: line;	

		file arquivo: text open read_mode is "entradasReais/entradasBin_10inputs.txt";
		variable linha: line;
		variable amostra: string(1 to 16);
		variable amostraTesteReal: real;
		variable hexvalUnsigned : unsigned(63 downto 0);
		variable hexvalSigned : signed(63 downto 0);
		variable bitvec : std_logic_vector(63 downto 0);

		begin

			if (rising_edge(clk)) then

				if not(endfile(arquivo)) then		-- enquanto exixtirem linhas no arquivo atual

					readline(arquivo, linha);	-- lê uma linha e armazena em uma string/line (linha)
					-- lê os valores da linhas e passa para as entradas do circuito
						read(linha, amostra);
							--entrada <= string_to_float16(amostra);
							entrada <= std_logic_vector(to_unsigned(amostra, 16));
						read(linha, amostra);
							saidaEsperada <= string_to_float16(amostra);
				end if;

			end if;

	end process EntradasProcess;

end testbench_arc;