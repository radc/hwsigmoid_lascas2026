library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use std.textio.all;
--use work.myTypes.all;
use work.float_pkg.all;
use work.LUT.all;

entity tb_wsilu_L is
end entity;

architecture sim of tb_wsilu_L is

    -- Declaração do componente a ser testado (DUT - Device Under Test)
    component wsilu_LUT_512
        port (
            clk    : in  std_logic;
            reset   : in  std_logic;
            enable : in  std_logic;
            xIn      : in  float16;
            yOut      : out float16
        );
    end component;

    -- Sinais para conectar ao DUT
    signal clk    : std_logic := '0';
    signal rst    : std_logic := '1';
    signal enable : std_logic := '0';
    signal x      : float16   := to_float(0.0, 5, 10);
    signal y      : float16;

    -- Constantes do clock
    constant clk_period : time := 20 ns;

begin

    -- Instanciação do DUT
    uut: wsilu_LUT_512
        port map (
            clk    => clk,
            reset    => rst,
            enable => enable,
            xIn      => x,
            yOut      => y
        );

    -- Processo para gerar o clock (mantido por boas práticas)
    clk_process : process
    begin
        while true loop
            clk <= '0';
            wait for clk_period / 2;
            clk <= '1';
            wait for clk_period / 2;
        end loop;
    end process;

    -- Processo de estímulo e verificação
    stimulus_process : process
        variable real_input            : real;
        variable sigmoid_real          : real;
        variable expected_wsilu_real   : real;
        variable dut_output_as_real    : real;
        variable error                 : real;
        variable output_line           : line;
    begin
        report ">> Iniciando Testbench para a WSiLU (Arquitetura Combinacional) <<";
        
        -- Aplica e libera o reset
        rst <= '1';
         wait for clk_period;
        rst <= '0';
        enable <= '1'; -- Habilita o componente

        -- Loop para varrer a entrada de -3.1 a 3.1, com passo de 0.1
        for i in -31 to 22 loop
            real_input := real(i) * 0.1;

            -- Modelo de referência ("Golden Model") para o valor esperado
            sigmoid_real := 1.0 / (1.0 + exp(-4.0 * real_input));
            expected_wsilu_real := real_input * sigmoid_real;

            -- 1. Aplica o estímulo na entrada 'x'
            x <= to_float(real_input, 5, 10);

            -- 2. Espera um tempo de propagação (a lógica é combinacional)
            -- Não é preciso esperar o clock, um pequeno delay é suficiente.
            wait for 2*clk_period;

            -- 3. Lê a saída do DUT
            dut_output_as_real := to_real(y);

            -- 4. Calcula o erro absoluto
            error := abs(expected_wsilu_real - dut_output_as_real);

            -- 5. Imprime os resultados no console
            write(output_line, string'(">> Testando x = "));
            write(output_line, real_input);
            writeline(output, output_line);

            write(output_line, string'("   - Saida do DUT (Aprox) : "));
            write(output_line, dut_output_as_real);
            writeline(output, output_line);

            write(output_line, string'("   - Saida Esperada (Ref) : "));
            write(output_line, expected_wsilu_real);
            writeline(output, output_line);

            write(output_line, string'("   - Erro Absoluto        : "));
            write(output_line, error);
            writeline(output, output_line);
            
            -- Espera o resto do ciclo para manter um ritmo de teste consistente
            wait for clk_period - 1 ns;

        end loop;

        report ">> Testbench concluído. <<";
        wait; -- Fim da simulação
    end process;

end architecture;
