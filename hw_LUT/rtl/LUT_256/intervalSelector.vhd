library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.float_pkg.all;
--use work.myTypes.all;
use work.fixed_pkg.all;
use work.float_pkg.all;
use work.LUT.all;

entity intervalSelector is 
    port(
        x        : in  float16;
        interval: out std_logic_vector(2 downto 0)
    );
end entity;

architecture arq of intervalSelector is

    constant limite1 : float16 := to_float(-2.0, 5, 10);
    constant limite2 : float16 := to_float(0.0, 5, 10);
    constant limite3 : float16 := to_float(0.5, 5, 10);
    constant limite4 : float16 := to_float(1.0, 5, 10);
    constant limite5: float16 := to_float(1.5, 5, 10);
    constant limite6 : float16 := to_float(2.0, 5, 10);
    begin
interval <= "000" when x < limite1 else
                 "101" when x > limite6 else
                 "001" when abs(x) >= limite2 and abs(x) <= limite3 else
                 "010" when abs(x) > limite3 and abs(x) <= limite4 else
                 "011" when abs(x) > limite4 and abs(x) <= limite5 else
                 "100"; -- Caso restante, onde abs(x) <= 2.0
end architecture;
