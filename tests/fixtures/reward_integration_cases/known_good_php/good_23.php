<?php
/**
 * Settings Page Handler 23
 *
 * @package WPFinetune\\Tests\\Fixtures\\KnownGood
 */

namespace WPFinetune\\Tests\\Fixtures\\KnownGood;

if ( ! defined( 'ABSPATH' ) ) {
	exit;
}

/**
 * Manages plugin settings page.
 */
class Settings_Handler_23 {{

	/**
	 * @var string Options key.
	 */
	const OPTION_KEY = 'wpf_settings_23';

	/**
	 * @var string Nonce action.
	 */
	const NONCE_ACTION = 'wpf_save_settings_23';

	/**
	 * Register settings and hooks.
	 *
	 * @return void
	 */
	public function init(): void {{
		add_action( 'admin_menu', array( $this, 'add_menu_page' ) );
		add_action( 'admin_init', array( $this, 'register_settings' ) );
	}}

	/**
	 * Add the menu page.
	 *
	 * @return void
	 */
	public function add_menu_page(): void {{
		add_options_page(
			__( 'WPF Settings 23', 'wpf-tests' ),
			__( 'WPF 23', 'wpf-tests' ),
			'manage_options',
			'wpf-settings-23',
			array( $this, 'render_page' )
		);
	}}

	/**
	 * Register settings.
	 *
	 * @return void
	 */
	public function register_settings(): void {{
		register_setting(
			'wpf_settings_group_23',
			self::OPTION_KEY,
			array(
				'sanitize_callback' => array( $this, 'sanitize_options' ),
				'default'          => array(),
			)
		);
	}}

	/**
	 * Sanitize options before saving.
	 *
	 * @param array $options Raw options array.
	 * @return array Sanitized options.
	 */
	public function sanitize_options( array $options ): array {{
		if ( ! current_user_can( 'manage_options' ) ) {{
			return get_option( self::OPTION_KEY, array() );
		}}
		$clean = array();
		if ( isset( $options['title'] ) ) {{
			$clean['title'] = sanitize_text_field( wp_unslash( $options['title'] ) );
		}}
		return $clean;
	}}

	/**
	 * Render the settings page.
	 *
	 * @return void
	 */
	public function render_page(): void {{
		if ( ! current_user_can( 'manage_options' ) ) {{
			return;
		}}
		$options = get_option( self::OPTION_KEY, array() );
		?>
		<div class="wrap">
			<h1><?php esc_html_e( 'Settings 23', 'wpf-tests' ); ?></h1>
			<form method="post" action="options.php">
				<?php
				settings_fields( 'wpf_settings_group_23' );
				wp_nonce_field( self::NONCE_ACTION );
				?>
				<table class="form-table">
					<tr>
						<th scope="row">
							<label for="wpf-title-23"><?php esc_html_e( 'Title', 'wpf-tests' ); ?></label>
						</th>
						<td>
							<input
								type="text"
								id="wpf-title-23"
								name="<?php echo esc_attr( self::OPTION_KEY ); ?>[title]"
								value="<?php echo esc_attr( $options['title'] ?? '' ); ?>"
							/>
						</td>
					</tr>
				</table>
				<?php submit_button(); ?>
			</form>
		</div>
		<?php
	}}
}}
