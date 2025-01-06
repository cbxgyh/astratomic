use crate::prelude::*;

fn setup() {
    puffin::set_scopes_on(true);
}

fn new_frame() {
    puffin::GlobalProfiler::lock().new_frame();
}

fn egui(mut egui_ctx: Query<&EguiContext, With<PrimaryWindow>>) {
    let Ok(mut ctx) = egui_ctx.get_single() else {
        return;
    };
    puffin_egui::profiler_window(ctx.get());
}

pub struct PuffinPlugin;
impl Plugin for PuffinPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(First, new_frame)
            .add_systems(Update, egui)
            .add_systems(Startup, setup);
    }
}
