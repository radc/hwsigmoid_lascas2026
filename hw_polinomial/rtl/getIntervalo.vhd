library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.float_pkg.all;

entity getIntervalo is 
    port(
        x        : in  float16;
        intervalo: out std_logic_vector(4 downto 0)
    );
end getIntervalo;

architecture getIntervalo_architecture of getIntervalo is

    constant limiteInferiorS1 : float16 := to_float(-2, 5, 10);
    constant limiteSuperiorS1 : float16 := to_float(-1.5, 5, 10);

    constant limiteInferiorS2 : float16 := to_float(-1.5, 5, 10);
    constant limiteSuperiorS2 : float16 := to_float(-1, 5, 10);

    constant limiteInferiorS3 : float16 := to_float(-1, 5, 10);
    constant limiteSuperiorS3 : float16 := to_float(-0.75, 5, 10);

    constant limiteInferiorS4 : float16 := to_float(-0.75, 5, 10);
    constant limiteSuperiorS4 : float16 := to_float(-0.5, 5, 10);

    constant limiteInferiorS5 : float16 := to_float(-0.5, 5, 10);
    constant limiteSuperiorS5 : float16 := to_float(-0.25, 5, 10);
    
    constant limiteInferiorS6 : float16 := to_float(-0.25, 5, 10);
    constant limiteSuperiorS6 : float16 := to_float(0, 5, 10);

    constant limiteInferiorS7 : float16 := to_float(0, 5, 10);
    constant limiteSuperiorS7 : float16 := to_float(0.25, 5, 10);
    
    constant limiteInferiorS8 : float16 := to_float(0.25, 5, 10);
    constant limiteSuperiorS8 : float16 := to_float(0.5, 5, 10);

    constant limiteInferiorS9 : float16 := to_float(0.5, 5, 10);
    constant limiteSuperiorS9 : float16 := to_float(0.75, 5, 10);
    
    constant limiteInferiorS10 : float16 := to_float(0.75, 5, 10);
    constant limiteSuperiorS10 : float16 := to_float(1, 5, 10);

    constant limiteInferiorS11 : float16 := to_float(1, 5, 10);
    constant limiteSuperiorS11 : float16 := to_float(1.25, 5, 10);
    
    constant limiteInferiorS12 : float16 := to_float(1.25, 5, 10);
    constant limiteSuperiorS12 : float16 := to_float(1.312, 5, 10);

    constant limiteInferiorS13 : float16 := to_float(1.312, 5, 10);
    constant limiteSuperiorS13 : float16 := to_float(1.375, 5, 10);
    
    constant limiteInferiorS14 : float16 := to_float(1.375, 5, 10);
    constant limiteSuperiorS14 : float16 := to_float(1.438, 5, 10);

    constant limiteInferiorS15 : float16 := to_float(1.438, 5, 10);
    constant limiteSuperiorS15 : float16 := to_float(1.5, 5, 10);
    
    constant limiteInferiorS16 : float16 := to_float(1.5, 5, 10);
    constant limiteSuperiorS16 : float16 := to_float(2, 5, 10);

begin

    intervalo <=    "00000" when x <= limiteInferiorS1 else
                    "00001" when x > limiteInferiorS1 and x <= limiteSuperiorS1 else
                    "00010" when x > limiteInferiorS2 and x <= limiteSuperiorS2 else
                    "00011" when x > limiteInferiorS3 and x <= limiteSuperiorS3 else
                    "00100" when x > limiteInferiorS4 and x <= limiteSuperiorS4 else
                    "00101" when x > limiteInferiorS5 and x <= limiteSuperiorS5 else
                    "00110" when x > limiteInferiorS6 and x <= limiteSuperiorS6 else
                    "00111" when x > limiteInferiorS7 and x <= limiteSuperiorS7 else
                    "01000" when x > limiteInferiorS8 and x <= limiteSuperiorS8 else
                    "01001" when x > limiteInferiorS9 and x <= limiteSuperiorS9 else
                    "01010" when x > limiteInferiorS10 and x <= limiteSuperiorS10 else
                    "01011" when x > limiteInferiorS11 and x <= limiteSuperiorS11 else
                    "01100" when x > limiteInferiorS12 and x <= limiteSuperiorS12 else
                    "01101" when x > limiteInferiorS13 and x <= limiteSuperiorS13 else
                    "01110" when x > limiteInferiorS14 and x <= limiteSuperiorS14 else
                    "01111" when x > limiteInferiorS15 and x <= limiteSuperiorS15 else
                    "10000" when x > limiteInferiorS16 and x <= limiteSuperiorS16 else
                    "11111" when x > limiteSuperiorS16;
    
end architecture getIntervalo_architecture;

