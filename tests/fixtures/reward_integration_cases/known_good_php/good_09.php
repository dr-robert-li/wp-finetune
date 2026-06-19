<?php
/**
 * Database Query Helper 9
 *
 * @package WPFinetune\\Tests\\Fixtures\\KnownGood
 */

namespace WPFinetune\\Tests\\Fixtures\\KnownGood;

if ( ! defined( 'ABSPATH' ) ) {
	exit;
}

/**
 * Query helper with prepared statements.
 */
class DB_Query_Helper_09 {{

	/**
	 * @var string Table name (without prefix).
	 */
	private string $table_base;

	/**
	 * Constructor.
	 *
	 * @param string $table_base Base table name.
	 */
	public function __construct( string $table_base ) {{
		$this->table_base = sanitize_key( $table_base );
	}}

	/**
	 * Get table name with prefix.
	 *
	 * @return string Full table name.
	 */
	private function get_table_name(): string {{
		global $wpdb;
		return $wpdb->prefix . $this->table_base;
	}}

	/**
	 * Find a record by ID.
	 *
	 * @param int $id Record ID.
	 * @return object|null Row object or null.
	 */
	public function find_by_id( int $id ): ?object {{
		global $wpdb;
		$table = $this->get_table_name();
		// phpcs:ignore WordPress.DB.DirectDatabaseQuery.DirectQuery
		return $wpdb->get_row(
			$wpdb->prepare( 'SELECT * FROM `%i` WHERE `id` = %d', $table, $id )
		);
	}}

	/**
	 * Insert a new record.
	 *
	 * @param array $data Associative array of column => value pairs.
	 * @return int|false Inserted row ID or false on failure.
	 */
	public function insert( array $data ): int|false {{
		global $wpdb;
		// phpcs:ignore WordPress.DB.DirectDatabaseQuery.DirectQuery
		$result = $wpdb->insert( $this->get_table_name(), $data );
		if ( false === $result ) {{
			return false;
		}}
		return (int) $wpdb->insert_id;
	}}

	/**
	 * Delete a record by ID.
	 *
	 * @param int $id Record ID.
	 * @return bool True on success, false on failure.
	 */
	public function delete_by_id( int $id ): bool {{
		if ( ! current_user_can( 'delete_posts' ) ) {{
			return false;
		}}
		global $wpdb;
		// phpcs:ignore WordPress.DB.DirectDatabaseQuery.DirectQuery
		$result = $wpdb->delete( $this->get_table_name(), array( 'id' => $id ), array( '%d' ) );
		return false !== $result;
	}}
}}
