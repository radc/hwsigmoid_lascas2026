library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.float_pkg.all;

entity getIntervalo is 
    port(
        x        : in  float32;
        intervalo: out std_logic_vector(3 downto 0)
    );
end getIntervalo;

architecture arquitetura of getIntervalo is

    constant limiteInferiorS1 : float32 := to_float(-2, 8, 23);
    constant limiteSuperiorS1 : float32 := to_float(-1, 8, 23);
    constant limiteInferiorS2 : float32 := to_float(-1, 8, 23);
    constant limiteSuperiorS2 : float32 := to_float(0, 8, 23);
    
begin

    intervalo <= "0000" when x > limiteInferiorS1 and x <= limiteSuperiorS1 else
                 "1111" when x > limiteInferiorS2 and x <= limiteSuperiorS2;
    
end architecture arquitetura;
