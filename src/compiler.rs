use crate::error::HasherError;

pub fn compile_glsl_to_spirv(
    name: &str,
    source: &str,
    entry_point: &str,
    macros: Option<Vec<(&str, Option<&str>)>>,
) -> Result<Vec<u32>, HasherError> {
    let compiler = shaderc::Compiler::new()?;

    let mut compile_options = shaderc::CompileOptions::new()?;

    compile_options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_1 as u32,
    );

    compile_options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    if let Some(macros) = macros {
        for (name, value) in macros {
            compile_options.add_macro_definition(name, value);
        }
    }

    let compilation_artifact = compiler.compile_into_spirv(
        source,
        shaderc::ShaderKind::Compute,
        name,
        entry_point,
        Some(&compile_options),
    )?;

    Ok(compilation_artifact.as_binary().into())
}
