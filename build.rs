use std::io::Write;

fn main() {
    println!("cargo::rerun-if-changed=src/shaders/hasher.glsl");

    let compiler = shaderc::Compiler::new().expect("Couldn't create shaderc compiler");

    let name = "hasher.glsl";
    let source = include_str!("src/shaders/hasher.glsl");
    let entry_point = "main";

    let mut compile_options =
        shaderc::CompileOptions::new().expect("Couldn't create shaderc compile options");

    compile_options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_1 as u32,
    );

    compile_options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    // TODO: Figure out why wgpu can't validate the shader with an unpack instruction
    // compile_options.add_macro_definition("USE_UNPACK_UINT_2X32", None);

    let compilation_artifact = compiler
        .compile_into_spirv(
            source,
            shaderc::ShaderKind::Compute,
            name,
            entry_point,
            Some(&compile_options),
        )
        .expect("Couldn't compile the shader");

    let mut f = std::fs::File::create("src/shaders/hasher.spv")
        .expect("Couldn't create file for the compiled shader");

    f.write_all(compilation_artifact.as_binary_u8())
        .expect("Couldn't write file for the compiled shader");
}
