library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.myTypes.all;
use work.fixed_pkg.all;
use work.float_pkg.all;

entity pwConvolution is --PointWise Convolution para mais um filtro
  port (
    clk    : in std_logic;
    reset  : in std_logic;
    enable : in std_logic;
     -- Entrada de 1 pixel da imagem de cada canal (256)
pIn_c0, pIn_c1, pIn_c2, pIn_c3, pIn_c4, pIn_c5, pIn_c6, pIn_c7, pIn_c8, pIn_c9, pIn_c10, pIn_c11, pIn_c12, pIn_c13, pIn_c14, pIn_c15, pIn_c16, pIn_c17, pIn_c18, pIn_c19, pIn_c20, pIn_c21, pIn_c22, pIn_c23, pIn_c24, pIn_c25, pIn_c26, pIn_c27, pIn_c28, pIn_c29, pIn_c30, pIn_c31, pIn_c32, pIn_c33, pIn_c34, pIn_c35, pIn_c36, pIn_c37, pIn_c38, pIn_c39, pIn_c40, pIn_c41, pIn_c42, pIn_c43, pIn_c44, pIn_c45, pIn_c46, pIn_c47, pIn_c48, pIn_c49, pIn_c50, pIn_c51, pIn_c52, pIn_c53, pIn_c54, pIn_c55, pIn_c56, pIn_c57, pIn_c58, pIn_c59, pIn_c60, pIn_c61, pIn_c62, pIn_c63, pIn_c64, pIn_c65, pIn_c66, pIn_c67, pIn_c68, pIn_c69, pIn_c70, pIn_c71, pIn_c72, pIn_c73, pIn_c74, pIn_c75, pIn_c76, pIn_c77, pIn_c78, pIn_c79, pIn_c80, pIn_c81, pIn_c82, pIn_c83, pIn_c84, pIn_c85, pIn_c86, pIn_c87, pIn_c88, pIn_c89, pIn_c90, pIn_c91, pIn_c92, pIn_c93, pIn_c94, pIn_c95, pIn_c96, pIn_c97, pIn_c98, pIn_c99, pIn_c100, pIn_c101, pIn_c102, pIn_c103, pIn_c104, pIn_c105, pIn_c106, pIn_c107, pIn_c108, pIn_c109, pIn_c110, pIn_c111, pIn_c112, pIn_c113, pIn_c114, pIn_c115, pIn_c116, pIn_c117, pIn_c118, pIn_c119, pIn_c120, pIn_c121, pIn_c122, pIn_c123, pIn_c124, pIn_c125, pIn_c126, pIn_c127, pIn_c128, pIn_c129, pIn_c130, pIn_c131, pIn_c132, pIn_c133, pIn_c134, pIn_c135, pIn_c136, pIn_c137, pIn_c138, pIn_c139, pIn_c140, pIn_c141, pIn_c142, pIn_c143, pIn_c144, pIn_c145, pIn_c146, pIn_c147, pIn_c148, pIn_c149, pIn_c150, pIn_c151, pIn_c152, pIn_c153, pIn_c154, pIn_c155, pIn_c156, pIn_c157, pIn_c158, pIn_c159, pIn_c160, pIn_c161, pIn_c162, pIn_c163, pIn_c164, pIn_c165, pIn_c166, pIn_c167, pIn_c168, pIn_c169, pIn_c170, pIn_c171, pIn_c172, pIn_c173, pIn_c174, pIn_c175, pIn_c176, pIn_c177, pIn_c178, pIn_c179, pIn_c180, pIn_c181, pIn_c182, pIn_c183, pIn_c184, pIn_c185, pIn_c186, pIn_c187, pIn_c188, pIn_c189, pIn_c190, pIn_c191, pIn_c192, pIn_c193, pIn_c194, pIn_c195, pIn_c196, pIn_c197, pIn_c198, pIn_c199, pIn_c200, pIn_c201, pIn_c202, pIn_c203, pIn_c204, pIn_c205, pIn_c206, pIn_c207, pIn_c208, pIn_c209, pIn_c210, pIn_c211, pIn_c212, pIn_c213, pIn_c214, pIn_c215, pIn_c216, pIn_c217, pIn_c218, pIn_c219, pIn_c220, pIn_c221, pIn_c222, pIn_c223, pIn_c224, pIn_c225, pIn_c226, pIn_c227, pIn_c228, pIn_c229, pIn_c230, pIn_c231, pIn_c232, pIn_c233, pIn_c234, pIn_c235, pIn_c236, pIn_c237, pIn_c238, pIn_c239, pIn_c240, pIn_c241, pIn_c242, pIn_c243, pIn_c244, pIn_c245, pIn_c246, pIn_c247, pIn_c248, pIn_c249, pIn_c250, pIn_c251, pIn_c252, pIn_c253, pIn_c254, pIn_c255: in float16;

    -- kernel 1x1 256 canais
kIn_c0, kIn_c1, kIn_c2, kIn_c3, kIn_c4, kIn_c5, kIn_c6, kIn_c7, kIn_c8, kIn_c9, kIn_c10, kIn_c11, kIn_c12, kIn_c13, kIn_c14, kIn_c15, kIn_c16, kIn_c17, kIn_c18, kIn_c19, kIn_c20, kIn_c21, kIn_c22, kIn_c23, kIn_c24, kIn_c25, kIn_c26, kIn_c27, kIn_c28, kIn_c29, kIn_c30, kIn_c31, kIn_c32, kIn_c33, kIn_c34, kIn_c35, kIn_c36, kIn_c37, kIn_c38, kIn_c39, kIn_c40, kIn_c41, kIn_c42, kIn_c43, kIn_c44, kIn_c45, kIn_c46, kIn_c47, kIn_c48, kIn_c49, kIn_c50, kIn_c51, kIn_c52, kIn_c53, kIn_c54, kIn_c55, kIn_c56, kIn_c57, kIn_c58, kIn_c59, kIn_c60, kIn_c61, kIn_c62, kIn_c63, kIn_c64, kIn_c65, kIn_c66, kIn_c67, kIn_c68, kIn_c69, kIn_c70, kIn_c71, kIn_c72, kIn_c73, kIn_c74, kIn_c75, kIn_c76, kIn_c77, kIn_c78, kIn_c79, kIn_c80, kIn_c81, kIn_c82, kIn_c83, kIn_c84, kIn_c85, kIn_c86, kIn_c87, kIn_c88, kIn_c89, kIn_c90, kIn_c91, kIn_c92, kIn_c93, kIn_c94, kIn_c95, kIn_c96, kIn_c97, kIn_c98, kIn_c99, kIn_c100, kIn_c101, kIn_c102, kIn_c103, kIn_c104, kIn_c105, kIn_c106, kIn_c107, kIn_c108, kIn_c109, kIn_c110, kIn_c111, kIn_c112, kIn_c113, kIn_c114, kIn_c115, kIn_c116, kIn_c117, kIn_c118, kIn_c119, kIn_c120, kIn_c121, kIn_c122, kIn_c123, kIn_c124, kIn_c125, kIn_c126, kIn_c127, kIn_c128, kIn_c129, kIn_c130, kIn_c131, kIn_c132, kIn_c133, kIn_c134, kIn_c135, kIn_c136, kIn_c137, kIn_c138, kIn_c139, kIn_c140, kIn_c141, kIn_c142, kIn_c143, kIn_c144, kIn_c145, kIn_c146, kIn_c147, kIn_c148, kIn_c149, kIn_c150, kIn_c151, kIn_c152, kIn_c153, kIn_c154, kIn_c155, kIn_c156, kIn_c157, kIn_c158, kIn_c159, kIn_c160, kIn_c161, kIn_c162, kIn_c163, kIn_c164, kIn_c165, kIn_c166, kIn_c167, kIn_c168, kIn_c169, kIn_c170, kIn_c171, kIn_c172, kIn_c173, kIn_c174, kIn_c175, kIn_c176, kIn_c177, kIn_c178, kIn_c179, kIn_c180, kIn_c181, kIn_c182, kIn_c183, kIn_c184, kIn_c185, kIn_c186, kIn_c187, kIn_c188, kIn_c189, kIn_c190, kIn_c191, kIn_c192, kIn_c193, kIn_c194, kIn_c195, kIn_c196, kIn_c197, kIn_c198, kIn_c199, kIn_c200, kIn_c201, kIn_c202, kIn_c203, kIn_c204, kIn_c205, kIn_c206, kIn_c207, kIn_c208, kIn_c209, kIn_c210, kIn_c211, kIn_c212, kIn_c213, kIn_c214, kIn_c215, kIn_c216, kIn_c217, kIn_c218, kIn_c219, kIn_c220, kIn_c221, kIn_c222, kIn_c223, kIn_c224, kIn_c225, kIn_c226, kIn_c227, kIn_c228, kIn_c229, kIn_c230, kIn_c231, kIn_c232, kIn_c233, kIn_c234, kIn_c235, kIn_c236, kIn_c237, kIn_c238, kIn_c239, kIn_c240, kIn_c241, kIn_c242, kIn_c243, kIn_c244, kIn_c245, kIn_c246, kIn_c247, kIn_c248, kIn_c249, kIn_c250, kIn_c251, kIn_c252, kIn_c253, kIn_c254, kIn_c255: in float16;
-- Saida
	pOut : out float16
  );
end entity;

architecture rtl of pwConvolution is

component pwconv_unit is 
  port (
      clk: in std_logic;
      reset:in std_logic;
      enable: in std_logic;
	pIn : in array_256_slv;
    -- kernel 1x1 256 canais
	kIn : in array_256_slv;
    	conv_out : out float16
  );
end component;
  signal  p, k,pIn, kIn : array_256_slv;
  signal cout : float16;

begin

p <= (pIn_c0, pIn_c1, pIn_c2, pIn_c3, pIn_c4, pIn_c5, pIn_c6, pIn_c7, pIn_c8, pIn_c9, pIn_c10, pIn_c11, pIn_c12, pIn_c13, pIn_c14, pIn_c15, pIn_c16, pIn_c17, pIn_c18, pIn_c19, pIn_c20, pIn_c21, pIn_c22, pIn_c23, pIn_c24, pIn_c25, pIn_c26, pIn_c27, pIn_c28, pIn_c29, pIn_c30, pIn_c31, pIn_c32, pIn_c33, pIn_c34, pIn_c35, pIn_c36, pIn_c37, pIn_c38, pIn_c39, pIn_c40, pIn_c41, pIn_c42, pIn_c43, pIn_c44, pIn_c45, pIn_c46, pIn_c47, pIn_c48, pIn_c49, pIn_c50, pIn_c51, pIn_c52, pIn_c53, pIn_c54, pIn_c55, pIn_c56, pIn_c57, pIn_c58, pIn_c59, pIn_c60, pIn_c61, pIn_c62, pIn_c63, pIn_c64, pIn_c65, pIn_c66, pIn_c67, pIn_c68, pIn_c69, pIn_c70, pIn_c71, pIn_c72, pIn_c73, pIn_c74, pIn_c75, pIn_c76, pIn_c77, pIn_c78, pIn_c79, pIn_c80, pIn_c81, pIn_c82, pIn_c83, pIn_c84, pIn_c85, pIn_c86, pIn_c87, pIn_c88, pIn_c89, pIn_c90, pIn_c91, pIn_c92, pIn_c93, pIn_c94, pIn_c95, pIn_c96, pIn_c97, pIn_c98, pIn_c99, pIn_c100, pIn_c101, pIn_c102, pIn_c103, pIn_c104, pIn_c105, pIn_c106, pIn_c107, pIn_c108, pIn_c109, pIn_c110, pIn_c111, pIn_c112, pIn_c113, pIn_c114, pIn_c115, pIn_c116, pIn_c117, pIn_c118, pIn_c119, pIn_c120, pIn_c121, pIn_c122, pIn_c123, pIn_c124, pIn_c125, pIn_c126, pIn_c127, pIn_c128, pIn_c129, pIn_c130, pIn_c131, pIn_c132, pIn_c133, pIn_c134, pIn_c135, pIn_c136, pIn_c137, pIn_c138, pIn_c139, pIn_c140, pIn_c141, pIn_c142, pIn_c143, pIn_c144, pIn_c145, pIn_c146, pIn_c147, pIn_c148, pIn_c149, pIn_c150, pIn_c151, pIn_c152, pIn_c153, pIn_c154, pIn_c155, pIn_c156, pIn_c157, pIn_c158, pIn_c159, pIn_c160, pIn_c161, pIn_c162, pIn_c163, pIn_c164, pIn_c165, pIn_c166, pIn_c167, pIn_c168, pIn_c169, pIn_c170, pIn_c171, pIn_c172, pIn_c173, pIn_c174, pIn_c175, pIn_c176, pIn_c177, pIn_c178, pIn_c179, pIn_c180, pIn_c181, pIn_c182, pIn_c183, pIn_c184, pIn_c185, pIn_c186, pIn_c187, pIn_c188, pIn_c189, pIn_c190, pIn_c191, pIn_c192, pIn_c193, pIn_c194, pIn_c195, pIn_c196, pIn_c197, pIn_c198, pIn_c199, pIn_c200, pIn_c201, pIn_c202, pIn_c203, pIn_c204, pIn_c205, pIn_c206, pIn_c207, pIn_c208, pIn_c209, pIn_c210, pIn_c211, pIn_c212, pIn_c213, pIn_c214, pIn_c215, pIn_c216, pIn_c217, pIn_c218, pIn_c219, pIn_c220, pIn_c221, pIn_c222, pIn_c223, pIn_c224, pIn_c225, pIn_c226, pIn_c227, pIn_c228, pIn_c229, pIn_c230, pIn_c231, pIn_c232, pIn_c233, pIn_c234, pIn_c235, pIn_c236, pIn_c237, pIn_c238, pIn_c239, pIn_c240, pIn_c241, pIn_c242, pIn_c243, pIn_c244, pIn_c245, pIn_c246, pIn_c247, pIn_c248, pIn_c249, pIn_c250, pIn_c251, pIn_c252, pIn_c253, pIn_c254, pIn_c255);

k<= (kIn_c0, kIn_c1, kIn_c2, kIn_c3, kIn_c4, kIn_c5, kIn_c6, kIn_c7, kIn_c8, kIn_c9, kIn_c10, kIn_c11, kIn_c12, kIn_c13, kIn_c14, kIn_c15, kIn_c16, kIn_c17, kIn_c18, kIn_c19, kIn_c20, kIn_c21, kIn_c22, kIn_c23, kIn_c24, kIn_c25, kIn_c26, kIn_c27, kIn_c28, kIn_c29, kIn_c30, kIn_c31, kIn_c32, kIn_c33, kIn_c34, kIn_c35, kIn_c36, kIn_c37, kIn_c38, kIn_c39, kIn_c40, kIn_c41, kIn_c42, kIn_c43, kIn_c44, kIn_c45, kIn_c46, kIn_c47, kIn_c48, kIn_c49, kIn_c50, kIn_c51, kIn_c52, kIn_c53, kIn_c54, kIn_c55, kIn_c56, kIn_c57, kIn_c58, kIn_c59, kIn_c60, kIn_c61, kIn_c62, kIn_c63, kIn_c64, kIn_c65, kIn_c66, kIn_c67, kIn_c68, kIn_c69, kIn_c70, kIn_c71, kIn_c72, kIn_c73, kIn_c74, kIn_c75, kIn_c76, kIn_c77, kIn_c78, kIn_c79, kIn_c80, kIn_c81, kIn_c82, kIn_c83, kIn_c84, kIn_c85, kIn_c86, kIn_c87, kIn_c88, kIn_c89, kIn_c90, kIn_c91, kIn_c92, kIn_c93, kIn_c94, kIn_c95, kIn_c96, kIn_c97, kIn_c98, kIn_c99, kIn_c100, kIn_c101, kIn_c102, kIn_c103, kIn_c104, kIn_c105, kIn_c106, kIn_c107, kIn_c108, kIn_c109, kIn_c110, kIn_c111, kIn_c112, kIn_c113, kIn_c114, kIn_c115, kIn_c116, kIn_c117, kIn_c118, kIn_c119, kIn_c120, kIn_c121, kIn_c122, kIn_c123, kIn_c124, kIn_c125, kIn_c126, kIn_c127, kIn_c128, kIn_c129, kIn_c130, kIn_c131, kIn_c132, kIn_c133, kIn_c134, kIn_c135, kIn_c136, kIn_c137, kIn_c138, kIn_c139, kIn_c140, kIn_c141, kIn_c142, kIn_c143, kIn_c144, kIn_c145, kIn_c146, kIn_c147, kIn_c148, kIn_c149, kIn_c150, kIn_c151, kIn_c152, kIn_c153, kIn_c154, kIn_c155, kIn_c156, kIn_c157, kIn_c158, kIn_c159, kIn_c160, kIn_c161, kIn_c162, kIn_c163, kIn_c164, kIn_c165, kIn_c166, kIn_c167, kIn_c168, kIn_c169, kIn_c170, kIn_c171, kIn_c172, kIn_c173, kIn_c174, kIn_c175, kIn_c176, kIn_c177, kIn_c178, kIn_c179, kIn_c180, kIn_c181, kIn_c182, kIn_c183, kIn_c184, kIn_c185, kIn_c186, kIn_c187, kIn_c188, kIn_c189, kIn_c190, kIn_c191, kIn_c192, kIn_c193, kIn_c194, kIn_c195, kIn_c196, kIn_c197, kIn_c198, kIn_c199, kIn_c200, kIn_c201, kIn_c202, kIn_c203, kIn_c204, kIn_c205, kIn_c206, kIn_c207, kIn_c208, kIn_c209, kIn_c210, kIn_c211, kIn_c212, kIn_c213, kIn_c214, kIn_c215, kIn_c216, kIn_c217, kIn_c218, kIn_c219, kIn_c220, kIn_c221, kIn_c222, kIn_c223, kIn_c224, kIn_c225, kIn_c226, kIn_c227, kIn_c228, kIn_c229, kIn_c230, kIn_c231, kIn_c232, kIn_c233, kIn_c234, kIn_c235, kIn_c236, kIn_c237, kIn_c238, kIn_c239, kIn_c240, kIn_c241, kIn_c242, kIn_c243, kIn_c244, kIn_c245, kIn_c246, kIn_c247, kIn_c248, kIn_c249, kIn_c250, kIn_c251, kIn_c252, kIn_c253, kIn_c254, kIn_c255);


pwconv_unit_inst : pwconv_unit
port map(
      clk=>clk,
      reset=> reset,
      enable=> enable,
	pIn => pIn,
	kIn => kIn,
  	conv_out => cout
);


inputReg: process (reset, clk)
	begin
		if (reset = '1') then
			pIn <= (others => (others => '0'));
			kIn <= (others => (others => '0'));
		elsif ((clk'event and clk = '1')) then
			pIn <= p;
			kIn <= k;
			pOut <= cout;
		end if;
	end process inputReg;

end architecture;


