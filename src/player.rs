use std::cmp::min;
use std::collections::VecDeque;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey,Material};
use bevy::render::mesh::{MeshVertexBufferLayout, PrimitiveTopology, VertexAttributeValues};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{AsBindGroup, PolygonMode, RenderPipelineDescriptor, ShaderRef, SpecializedMeshPipelineError};
use bevy::sprite::{Anchor, Material2d, Material2dKey, Material2dPlugin, MaterialMesh2dBundle, Mesh2dHandle};
use crate::prelude::*;

#[derive(Component)]
pub struct Player {
    fuel: f32,
    state: PlayerState,
    atom_id: u8,
}

impl Default for Player {
    fn default() -> Self {
        Self {
            fuel: FUEL_MAX,
            state: PlayerState::default(),
            atom_id: 2,
        }
    }
}

impl Drop for Actor {
    fn drop(&mut self) {
        let file = File::create("assets/world/player").unwrap();
        let mut buffered = BufWriter::new(file);
        bincode::serialize_into(&mut buffered, &self.pos).unwrap();
    }
}

#[derive(Default)]
pub enum PlayerState {
    #[default]
    Idle,
    Walking,
    Jumping(f64),
    Jetpack(bool),
}

#[derive(Component, Default)]
pub struct Tool;

#[derive(Component)]
pub struct ToolFront;

pub fn player_setup(
    mut commands: Commands,
    mut trail: ResMut<Trail>,
    mut materials: ResMut<Assets<LineMaterial>>,
    asset_server: Res<AssetServer>,
    mut texture_atlases: ResMut<Assets<TextureAtlasLayout>>,
    mut meshes: ResMut<Assets<Mesh>>
) {
    let pos: IVec2;
    if let Ok(file) = File::open("assets/world/player") {
        let mut buffered = BufReader::new(file);
        pos = bincode::deserialize_from(&mut buffered).unwrap();
    } else {
        pos = IVec2::default();
        let file = File::create("assets/world/player").unwrap();
        let mut buffered = BufWriter::new(file);
        bincode::serialize_into(&mut buffered, &pos).unwrap();
    }

    let player_actor = Actor {
        height: 17,
        width: 10,
        pos,
        vel: vec2(0., 0.),
    };

    let player_handle = asset_server.load("player/player_sheet.png");
    let animation_indices = AnimationIndices { first: 0, last: 1 };
    let player_atlas_layout =
        TextureAtlasLayout::from_grid(Vec2::new(24.0, 24.0), 8, 5, None, None);
    let atlas = TextureAtlas {
        index: animation_indices.first,
        layout: texture_atlases.add(player_atlas_layout),
    };
    let player_transform = GlobalTransform::from_xyz(5. * 3., -8. * 3., PLAYER_LAYER);

    let tool_handle = asset_server.load("player/player_tool.png");
    let tool_bundle = SpriteBundle {
        texture: tool_handle,
        sprite: Sprite {
            anchor: Anchor::Custom(vec2(-0.1, 0.)),
            ..Default::default()
        },
        transform: Transform::from_translation(Vec3::new(-1., -3.5, 0.1)),
        ..Default::default()
    };
    let tool_front_ent = commands
        .spawn((
            TransformBundle::from_transform(Transform::from_translation(vec3(5., 0., 0.))),
            ToolFront,
        ))
        .id();
    let tool_ent = commands
        .spawn(tool_bundle)
        .insert(Tool)
        .insert_children(0, &[tool_front_ent])
        .id();

    commands
        .spawn((
            player_actor.clone(),
            Player::default(),
            SpriteSheetBundle {
                atlas,
                global_transform: player_transform,
                texture: player_handle,
                ..default()
            },
            animation_indices,
            AnimationTimer(Timer::from_seconds(0.1, TimerMode::Repeating)),
            bevy_rapier2d::prelude::RigidBody::Fixed,
            bevy_rapier2d::prelude::LockedAxes::ROTATION_LOCKED,
            bevy_rapier2d::prelude::Collider::cuboid(
                player_actor.width as f32 / 2.,
                player_actor.height as f32 / 2.,
            ),
        ))
        .add_child(tool_ent);
    // 创建一个新的 Mesh Rectangle::default()
    let mut mesh = Mesh::new(PrimitiveTopology::LineList,RenderAssetUsages::RENDER_WORLD);
    let vertices = vec![
        // 顶点位置
        Vec3::new(-0.5, 0.0, 0.0), // 起点
        Vec3::new(0.5, 0.0, 0.0),  // 终点
        // 可以添加其他顶点以在线段上创建点
    ];

    let indices = vec![0, 1];
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::from(vertices));
    mesh.insert_indices(bevy::render::mesh::Indices::U32(indices));

    // 添加网格到场景
    commands.spawn(PbrBundle {
        mesh: meshes.add(mesh),
        ..Default::default()
    })
        .insert(PlayerTrail { positions: Vec::new() });
    ;
    println!("player_setup");

    {// 更新
        println!("生成line");
        let z= LineStrip {
            points: trail.create_Lines.iter().map(|pos|pos.xy()).collect()
        };
        let n: Mesh2dHandle = meshes.add(z).into();
        let entity= commands.spawn(
            MaterialMesh2dBundle {
                mesh: n,
                material: materials.add(LineMaterial{
                    color:Color::WHITE
                }),
                transform: Transform::from_translation(Vec3::new(-200., 0., 0.)),
                ..default()
            },

        ).id();
        commands.entity(entity).insert(crate::player::FlagLine{
            entity: Some(entity),
        });
    }
}
#[derive(Component)]
struct PlayerTrail {
    positions: Vec<Vec3>,
}
/// Updates player
pub fn update_player(
    input: (Res<Inputs>, EventReader<MouseWheel>),
    mut player: Query<(&mut Actor, &mut Player, &mut AnimationIndices)>,
    chunk_manager: ResMut<ChunkManager>,
    materials: (Res<Assets<Materials>>, Res<MaterialsHandle>),
    time: Res<Time>,
    mut zoom: ResMut<Zoom>,
) {
    let (mut actor, mut player, mut anim_idxs) = player.single_mut();
    let (inputs, mut scroll_evr) = input;
    let materials = materials.0.get(materials.1 .0.clone()).unwrap();

    // Gravity
    if actor.vel.y < TERM_VEL as f32 {
        actor.vel.y += 1.
            * if matches!(player.state, PlayerState::Jetpack { .. }) {
            0.4
        } else {
            1.
        };
    }

    // Movement
    let x = inputs.right - inputs.left;
    actor.vel.x = x * RUN_SPEED;

    let on_ground = on_ground(&chunk_manager, &actor, materials);

    // Refuel
    if on_ground {
        player.fuel = (player.fuel + FUEL_REGEN).clamp(0., Player::default().fuel);
    }

    if on_ground {
        if x.abs() > 0. {
            player.state = PlayerState::Walking
        } else {
            player.state = PlayerState::Idle
        }
    }
    // 跳跃按键刚按下时（jump_just_pressed），如果玩家站在地面上（on_ground），就会让玩家跳跃（actor.vel.y -= JUMP_MAG）。
    // 跳跃状态会被设为 Jumping 并记录跳跃的开始时间。
    // 在跳跃状态中，如果玩家仍然按着跳跃键（jump_pressed）并且按下的时间小于设定的最大时间（TIME_JUMP_PRESSED），
    // 则会增加跳跃的高度（actor.vel.y -= PRESSED_JUMP_MAG）。
    // Jump
    if inputs.jump_just_pressed {
        if on_ground {
            actor.vel.y -= JUMP_MAG;
            player.state = PlayerState::Jumping(time.elapsed_seconds_wrapped_f64());
        } else {
            player.state = PlayerState::Jetpack(true);
            actor.vel.y = 0.;
        }
    }

    //Jump higher when holding space
    if let PlayerState::Jumping(jump_start) = player.state {
        if inputs.jump_pressed
            && time.elapsed_seconds_wrapped_f64() - jump_start < TIME_JUMP_PRESSED
        {
            actor.vel.y -= PRESSED_JUMP_MAG
        }
    }

    // Jetpack
    let mut new_up = false;
    if let PlayerState::Jetpack(_) = player.state {
        if player.fuel > 0. && inputs.jump_pressed {
            actor.vel.y = (actor.vel.y - JETPACK_FORCE).clamp(-JETPACK_MAX, f32::MAX);
            player.fuel -= FUEL_COMSUMPTON;
            new_up = true;
        } else {
            new_up = false;
        }
    }

    if let PlayerState::Jetpack(up) = &mut player.state {
        *up = new_up
    };

    //Animation
    (anim_idxs.first, anim_idxs.last) = match player.state {
        PlayerState::Idle => (0, 1),
        PlayerState::Walking => (8, 11),
        PlayerState::Jumping { .. } => (16, 23),
        PlayerState::Jetpack(up) => {
            if up {
                (24, 26)
            } else {
                (32, 32)
            }
        }
    };

    //Zoom
    for ev in scroll_evr.read() {
        if ev.unit == MouseScrollUnit::Line {
            zoom.0 *= 0.9_f32.powi(ev.y as i32);
            zoom.0 = zoom.0.clamp(ZOOM_LOWER_BOUND, ZOOM_UPPER_BOUND);
        }
    }

    //Change shooting atoms
    if inputs.numbers[0] {
        player.atom_id = 2;
    } else if inputs.numbers[1] {
        player.atom_id = 3;
    } else if inputs.numbers[2] {
        player.atom_id = 4;
    } else if inputs.numbers[3] {
        player.atom_id = 5;
    }
}

pub fn tool_system(
    mut commands: Commands,
    mut tool: Query<(&mut Transform, &GlobalTransform, &mut Sprite), With<Tool>>,
    mut camera: Query<(&Camera, &GlobalTransform), Without<Tool>>,
    tool_front_ent: Query<Entity, With<ToolFront>>,
    querys: (Query<&Window>, Query<(&mut Sprite, &Player), Without<Tool>>),
    resources: (ResMut<ChunkManager>, ResMut<DirtyRects>, Res<Inputs>),
    materials: (Res<Assets<Materials>>, Res<MaterialsHandle>),
) {
    let (mut tool_transform, tool_gtransform, mut tool_sprite) = tool.single_mut();
    let (camera, camera_gtransform) = camera.single_mut();
    let (window, mut player) = querys;
    let (mut textatlas_sprite, player) = player.single_mut();
    let (mut chunk_manager, mut dirty_rects, inputs) = resources;
    let Ok(window) = window.get_single() else {
        return;
    };
    let materials = materials.0.get(materials.1 .0.clone()).unwrap();

    if let Some(world_position) = window
        .cursor_position()
        .and_then(|cursor| camera.viewport_to_world(camera_gtransform, cursor))
        .map(|ray| ray.origin.truncate())
    {
        //Rotate and move sprite
        let center_vec = tool_gtransform.compute_transform().translation.xy();
        let tool_vec = world_position - center_vec;
        let angle = tool_vec.y.atan2(tool_vec.x);
        tool_transform.rotation = Quat::from_rotation_z(angle);

        let flip_bool = angle.abs() > std::f32::consts::FRAC_PI_2;
        textatlas_sprite.flip_x = flip_bool;
        tool_sprite.flip_y = flip_bool;
        tool_transform.translation.x =
            tool_transform.translation.x.abs() * (flip_bool as i8 * 2 - 1) as f32;

        //Tool pulling and pushing atoms
        let mut center_vec_y_flipped = center_vec;
        center_vec_y_flipped.y *= -1.;

        let tool_slope = Vec2::new(angle.cos(), -angle.sin());
        let bound_slope = Vec2::new((angle + std::f32::consts::FRAC_PI_2).cos(), -(angle).cos());
        let tool_front = center_vec_y_flipped + tool_slope * 5.;

        let mut pos_to_update = vec![];
        if inputs.push {
            let new_tool_front = tool_front + tool_slope * 3.5;
            let n = 6;

            for i in 0..=n {
                let rand_angle = fastrand::f32() * std::f32::consts::TAU;

                let mut vec = new_tool_front - bound_slope * 2.
                    + bound_slope * 2.5 * i as f32 / n as f32
                    + vec2(rand_angle.cos(), rand_angle.sin());

                vec += tool_slope * 7. * angle.sin().max(0.);

                let chunk_pos = global_to_chunk(vec.as_ivec2());
                if let (Some(atom), tool_atom) = (
                    chunk_manager.get_mut_atom(chunk_pos),
                    Atom::new(player.atom_id),
                ) {
                    if materials[atom.id].is_void() || materials[atom.id].is_object() {
                        let angle = fastrand::f32() * 0.5 - 0.25;
                        let vel = (tool_slope * 10. * (fastrand::f32() * 0.2 + 0.8))
                            .rotate(vec2(angle.cos(), angle.sin()));
                        commands.spawn(Particle {
                            atom: tool_atom,
                            velocity: vel,
                            pos: vec,
                            ..Default::default()
                        });
                    }
                }
            }
        } else if inputs.pull {
            let center_bound = tool_front + tool_slope * TOOL_DISTANCE;

            let bound1 = (center_bound + bound_slope * TOOL_RANGE).as_ivec2();
            let bound2 = (center_bound + -bound_slope * TOOL_RANGE).as_ivec2();

            for bound_vec in Line::new(bound1, bound2 - bound1) {
                for vec in Line::new(
                    (tool_front - 4. * tool_slope).as_ivec2(),
                    bound_vec - (tool_front - 4. * tool_slope).as_ivec2(),
                ) {
                    let chunk_pos = global_to_chunk(vec);
                    if (vec.distance_squared((tool_front - 6. * tool_slope).as_ivec2()) as f32)
                        .sqrt()
                        < 6.
                    {
                        continue;
                    }

                    if let Some(atom) = chunk_manager.get_mut_atom(chunk_pos) {
                        if !materials[atom.id].is_void() && !materials[atom.id].is_object() {
                            commands.spawn(Particle {
                                atom: *atom,
                                pos: chunk_pos.to_global().as_vec2(),
                                state: PartState::Follow(tool_front_ent.single()),
                                ..Default::default()
                            });

                            pos_to_update.push(chunk_pos);
                            *atom = Atom::default();
                            break;
                        }
                    }
                }
            }
        }

        let mut chunks = HashSet::new();
        for pos in pos_to_update {
            update_dirty_rects_3x3(&mut dirty_rects.current, pos);
            update_dirty_rects(&mut dirty_rects.render, pos);
            chunks.insert(pos.chunk);
        }

        for chunk in chunks {
            let chunk = chunk_manager.chunks.get(&chunk).unwrap();
            commands.entity(chunk.entity.unwrap()).remove::<Collider>();
        }
    }
}

pub fn update_player_sprite(
    mut query: Query<(&mut Transform, &Actor,&Player)>,
    mut trail: ResMut<Trail>,
) {
    let (mut transform, actor,player) = query.single_mut();
    let top_corner_vec = vec3(actor.pos.x as f32, -actor.pos.y as f32, 2.);
    let center_vec = top_corner_vec + vec3(actor.width as f32 / 2., -8., 0.);
    transform.translation = center_vec;

    let clo= match player.state {
        PlayerState::Jumping(_) => 0.9,
        PlayerState::Jetpack(_) => 0.8,
        PlayerState::Walking=> 0.7,
        _=> 1.0
    };

    if let Some(last)=trail.points.iter().last(){
        if last.xy()!=center_vec.xy() {
            trail.add_point(Vec3::new(center_vec.x,center_vec.y,clo));
        }
    }else {
        trail.add_point(Vec3::new(center_vec.x,center_vec.y,clo));
    }

}


#[derive(Resource, Default)]
pub struct SavingTask(pub Option<Task<()>>);

pub fn get_input(
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut inputs: ResMut<Inputs>,
) {
    //Jump
    if keys.just_pressed(KeyCode::Space) {
        inputs.jump_just_pressed = true;
        inputs.jump_pressed = true;
    } else if keys.pressed(KeyCode::Space) {
        inputs.jump_pressed = true;
    }

    //Movement
    if keys.pressed(KeyCode::KeyA) {
        inputs.left = 1.;
    }
    if keys.pressed(KeyCode::KeyD) {
        inputs.right = 1.;
    }

    //Tool
    if mouse_buttons.pressed(MouseButton::Left) {
        inputs.pull = true;
    }
    if mouse_buttons.pressed(MouseButton::Right) {
        inputs.push = true;
    }

    //Numbers, to select atoms
    if keys.just_pressed(KeyCode::Digit1) {
        inputs.numbers[0] = true;
    } else if keys.just_pressed(KeyCode::Digit2) {
        inputs.numbers[1] = true;
    } else if keys.just_pressed(KeyCode::Digit3) {
        inputs.numbers[2] = true;
    } else if keys.just_pressed(KeyCode::Digit4) {
        inputs.numbers[3] = true;
    }
}

pub fn clear_input(mut inputs: ResMut<Inputs>) {
    *inputs = Inputs::default();
}

#[derive(Resource, Default)]
pub struct Inputs {
    left: f32,
    right: f32,

    pull: bool,
    push: bool,

    jump_pressed: bool,
    jump_just_pressed: bool,

    numbers: [bool; 4],
}

#[derive(Resource)]
struct Trail {
    points: VecDeque<Vec3>,  // 使用 VecDeque 来存储轨迹点  xy 位置 z 颜色
    create_points: VecDeque<Vec3>,
    create_Lines:VecDeque<Vec3>,
    max_points: usize,        // 最大轨迹点数量
}

impl Default for Trail {
    fn default() -> Self {
        Trail {
            points: VecDeque::default(),  // 使用 VecDeque 来存储轨迹点
            create_points: VecDeque::default(),
            create_Lines: VecDeque::default(),
            max_points: 300,        // 最大轨迹点数量
        }
    }
}
impl Trail {
    fn add_point(&mut self, point: Vec3) {
        // println!("add_point_:{:?}",self.points.len());
        if self.points.len() < self.max_points {
            // self.remove_points.push_back(self.points.pop_front().unwrap());  // 删除最老的点
            self.create_points.push_back(point);  // 添加新点
        }else{
            self.points.pop_front();
            self.points.push_back(point);
        }

        // println!("add_point_back:{:?}",self.points.len());
    }
}
#[derive(Component)]
struct FlagLine{
    entity: Option<Entity>,
}
#[derive(Component)]
struct FlagPoint;
fn render_trail(
    mut trail: ResMut<Trail>,
    mut query_trail:Query<(&mut Transform,&mut Sprite),With<FlagPoint>>,
    mut commands: Commands) {
    let create_points= &trail.create_points;

    let max_points= trail.max_points/2;
    // 渲染轨迹点
    for (i, point) in create_points.iter().enumerate() {
        let a=  if 1>=max_points { 1.0}else{0.3};
        commands.spawn((
            SpriteBundle {
                transform: Transform::from_translation(Vec3::new(point.x, point.y, 0.0)),
                sprite: Sprite {
                    color: Color::rgba(1.0, 0.0, 0.0,a),  // 可以根据轨迹的老化程度调整颜色
                    ..Default::default()
                },
                ..Default::default()
            },FlagPoint));

    }
    // 先把 trail.create_points 的值赋给一个临时变量
    let mut create_points = std::mem::take(&mut trail.create_points);
    trail.create_Lines.append(&mut create_points.clone());
    // 然后对 trail.points 进行操作，使用刚才的临时变量
    trail.points.append(&mut create_points);
    // 如果需要，再把修改后的 create_points 赋值回 trail.create_points
    trail.create_points = create_points;

    query_trail.iter_mut().enumerate()
        .zip(trail.points.iter())
        .for_each(|((i,(mut tr,mut sp)), elem2)|{
            tr.translation=*elem2;
            if i>= trail.max_points/2 {
                sp.color.set_a( 0.3);
            }else{
                sp.color.set_a( 1.0);
            }
        });

}
fn render_trail_line(
    mut trail: ResMut<Trail>,
    mut materials: ResMut<Assets<LineMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    query_player: Query<(&Transform),With<Player>>,
    mut commands: Commands,
    mut query_trail_line:Query<(&mut Transform,&FlagLine),Without<Player>>,
){

    //  生成
    println!("render_trail_line_:{:?},____{:?}",query_trail_line.iter().len(),trail.points.len());
    for  (mut tr,fl) in query_trail_line.iter_mut(){
        let player_tr=query_player.get_single().unwrap();
        if let Some(e)=fl.entity{
            let x=if trail.points.len() == trail.max_points{
                trail.points.clone()
            }else {
                trail.create_Lines.clone()
            };
            let z= LineStrip {
                points: x.iter().map(|pos|pos.xy()).collect(),
            };
            let n: Mesh2dHandle = meshes.add(z).into();
            commands.entity(e).despawn();
            let eid=commands.spawn(
                    MaterialMesh2dBundle {
                        mesh: n,
                        material: materials.add(LineMaterial{
                            color:Color::WHITE
                        }),
                        transform: Transform::from_translation(player_tr.translation.xyz()),
                        ..default()
                    }
                ).id();
            commands.entity(eid).insert(FlagLine{entity:Some(eid)});
        }
    }
}





fn f32_color(f:f32,a:f32) -> Color{
    match f {
        0.9=>Color::rgba(1.0, 0.0, 0.0,a),
        0.8=>Color::rgba(0.0, 0.0, 1.0,a),
        0.7=>Color::rgba(0.0, 1.0, 0.0,a),
        _=>Color::rgba(1.0, 1.0, 1.0,a),
    }
}


#[derive(Asset, TypePath, Default, AsBindGroup, Debug, Clone)]
struct LineMaterial {
    #[uniform(0)]
    color: Color,
}

impl Material2d for LineMaterial {
    fn fragment_shader() -> ShaderRef {
        "line_material.wgsl".into()
    }

    fn specialize(
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayout,
        key: Material2dKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // This is the important part to tell bevy to render this material as a line between vertices
        descriptor.primitive.polygon_mode = PolygonMode::Line;
        Ok(())
    }
}
#[derive(Debug, Clone)]
struct LineList {
    lines: VecDeque<(Vec2, Vec2)>,
}

impl From<LineList> for Mesh {
    fn from(line: LineList) -> Self {
        let vertices: Vec<_> = line.lines.into_iter().flat_map(|(a, b)| [a.extend(0.), b.extend(0.)]).collect();

        Mesh::new(
            // This tells wgpu that the positions are list of lines
            // where every pair is a start and end point
            PrimitiveTopology::LineList,
            RenderAssetUsages::RENDER_WORLD,
        )
            // Add the vertices positions as an attribute
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
    }
}
#[derive(Debug, Clone)]
struct LineStrip {
    points: Vec<Vec2>,
}

impl From<LineStrip> for Mesh {
    fn from(line: LineStrip) -> Self {
        let x:Vec<Vec3>=line.points.into_iter()
            .map(|vec2| Vec3::new(vec2.x, vec2.y, 0.0)) // 这里将 z 设置为 0.0
            .collect();
        Mesh::new(
            // This tells wgpu that the positions are a list of points
            // where a line will be drawn between each consecutive point
            PrimitiveTopology::LineStrip,
            RenderAssetUsages::RENDER_WORLD,
        )
            // Add the point positions as an attribute
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, x)
    }
}


pub struct PlayerPlugin;
impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {

        app
            .add_systems(
            FixedUpdate,
            (
                update_player.before(update_actors),
                update_player_sprite.after(update_actors),
                render_trail.after(update_player_sprite),
                render_trail_line.after(render_trail),
                tool_system
                    .before(chunk_manager_update)
                    .before(update_particles),
                clear_input.after(update_player).after(tool_system),

            )
                .run_if(in_state(GameState::Game)),
        )
            // .add_systems(
            //     Update,
            //     ray_cast_system
            // )
            // .add_systems(PostUpdate,clean_line)
            .add_systems(PreUpdate, get_input.run_if(in_state(GameState::Game)))
            .init_resource::<SavingTask>()
            .init_resource::<Inputs>()
            .init_resource::<Trail>()
            .add_plugins(( Material2dPlugin::<LineMaterial>::default()))
            .add_systems(OnEnter(GameState::Game), player_setup.after(manager_setup));
    }
}