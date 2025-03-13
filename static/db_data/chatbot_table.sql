CREATE TABLE public.chat_sessions (
	id serial4 NOT NULL,
	session_id uuid NOT NULL,
	user_id int4 DEFAULT 1 NOT NULL,
	title varchar(255) NOT NULL,
	created_at timestamp DEFAULT now() NULL,
	updated_at timestamp DEFAULT now() NULL,
	CONSTRAINT chat_sessions_pk PRIMARY KEY (id),
	CONSTRAINT chat_sessions_uk1 UNIQUE (session_id)
);

CREATE TABLE public.chat_messages (
	id serial4 NOT NULL,
	session_id uuid NOT NULL,
	message_id uuid NOT NULL,
	user_message text NOT NULL,
	main_message text NOT NULL,
	think_message text NULL,
	think_time int4 NULL,
	liked bool NULL,
	disliked bool NULL,
	dislike_feedback text NULL,
	created_at timestamp DEFAULT now() NULL,
	updated_at timestamp DEFAULT now() NULL,
	CONSTRAINT chat_messages_pk PRIMARY KEY (id),
	CONSTRAINT chat_messages_uk1 UNIQUE (session_id, message_id)
);

ALTER TABLE public.chat_messages ADD CONSTRAINT chat_messages_fk1 FOREIGN KEY (session_id) REFERENCES public.chat_sessions(session_id) ON DELETE CASCADE;


CREATE TABLE image_sessions (
	id serial4 NOT NULL,
	session_id varchar(21) NOT NULL,
	user_id int4 DEFAULT 1 NOT NULL,
	title varchar(255) NOT NULL,
	created_at timestamp DEFAULT now() NULL,
	updated_at timestamp DEFAULT now() NULL,
	CONSTRAINT image_sessions_pk PRIMARY KEY (id),
	CONSTRAINT image_sessions_u1 UNIQUE (session_id)
);

CREATE TABLE public.image_messages (
	id serial4 NOT NULL,
	session_id varchar(21) NOT NULL,
	message_id varchar(21) NOT NULL,
	image_seq int4 NOT NULL,
	user_message text NOT NULL,
	image_prompt text NOT NULL,
	image_url TEXT NULL,
	liked bool NULL,
	disliked bool NULL,
	dislike_feedback text NULL,
	created_at timestamp DEFAULT now() NULL,
	updated_at timestamp DEFAULT now() NULL,
	CONSTRAINT image_messages_pk PRIMARY KEY (id),
	CONSTRAINT image_messages_uk1 UNIQUE (session_id, message_id, image_seq)
);


-- public.chat_messages foreign keys
ALTER TABLE public.image_messages ADD CONSTRAINT image_messages_fk1 FOREIGN KEY (session_id) REFERENCES public.image_sessions(session_id) ON DELETE CASCADE;